# Stable variant of core.function.train_adaptive (original left unchanged).
#
# Differences from the original, each controlled by config (defaults reproduce it):
#   TRAIN.ADV_LOSS
#     'neg_bce'   loss_p = MSE - LAMBDA * BCE(D(f), domain_label)   (original)
#                 Unbounded below: the pose net is rewarded for making D arbitrarily
#                 wrong, which destabilised training at LAMBDA 2.5e-3 (job 11568480).
#     'confusion' loss_p = MSE + LAMBDA * BCE(D(f), 0.5)
#                 Domain-confusion loss (Tzeng et al. 2015). Bounded: minimum ln 2 when
#                 D outputs 0.5 for every image; gradient vanishes at full confusion.
#   TRAIN.LOCK_D
#     False  the pose-step backward also writes gradients into D, which the next
#            D step then consumes (original behaviour)
#     True   D's parameters are frozen during the pose step (paper §IV-B Stage II),
#            so D is updated only by its own classification loss.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import time

import torch

from core.evaluate import accuracy
from core.function import AverageMeter
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def adversarial_term(config, criterion_d, domain_logits_p, domain_label):
    """Term added to the pose MSE. Returns (term, bce_vs_true_label)."""
    bce_true = criterion_d(domain_logits_p, domain_label)
    mode = config.TRAIN.ADV_LOSS
    if mode == 'neg_bce':
        term = -config.TRAIN.LAMBDA * bce_true
    elif mode == 'confusion':
        half = torch.full_like(domain_label, 0.5)
        term = config.TRAIN.LAMBDA * criterion_d(domain_logits_p, half)
    else:
        raise ValueError('unknown TRAIN.ADV_LOSS: {}'.format(mode))
    return term, bce_true


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


def train_adaptive(config, train_loader, model_p, model_d, criterion_p, criterion_d,
                   optimizer_p, optimizer_d, epoch, output_dir, tb_log_dir, writer_dict,
                   losses_P_list, losses_D_list, acces_P_list, acces_D_list,
                   acc_num_total, num, losses_p, acc_p, losses_d):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_d_2 = AverageMeter()
    losses_mse = AverageMeter()
    losses_adv = AverageMeter()

    if epoch == 0 or not hasattr(train_adaptive, '_announced'):
        logger.info('function_stable: ADV_LOSS={} LOCK_D={} LAMBDA={}'.format(
            config.TRAIN.ADV_LOSS, config.TRAIN.LOCK_D, config.TRAIN.LAMBDA))
        train_adaptive._announced = True

    model_d.train()
    model_p.train()
    end = time.time()

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        feature_outputs, outputs = model_p(input)

        ######## Step I: Domain Classifier Update ########
        domain_logits = model_d(feature_outputs.detach())
        domain_label = (meta['synthetic'].unsqueeze(-1) * 1.0).cuda()

        loss_d = criterion_d(domain_logits, domain_label)
        # No zero_grad here on purpose: with LOCK_D False the pose step's gradient
        # left in D is consumed now, exactly as in the original. With LOCK_D True
        # nothing is left in D, so this is D's own gradient only.
        loss_d.backward()
        optimizer_d.step()
        optimizer_d.zero_grad()

        ######## Step II: Pose_Net Update ########
        if config.TRAIN.LOCK_D:
            set_requires_grad(model_d, False)
        domain_logits_p = model_d(feature_outputs)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        loss_mse = criterion_p(outputs, target, target_weight)
        adv, loss_d_2 = adversarial_term(config, criterion_d, domain_logits_p, domain_label)
        loss_p = loss_mse + adv

        optimizer_p.zero_grad()
        loss_p.backward()
        optimizer_p.step()
        optimizer_p.zero_grad()
        if config.TRAIN.LOCK_D:
            set_requires_grad(model_d, True)

        losses_p.update(loss_p.item(), input.size(0))
        losses_d.update(loss_d.item(), input.size(0))
        losses_d_2.update(loss_d_2.item(), input.size(0))
        losses_mse.update(loss_mse.item(), input.size(0))
        losses_adv.update(adv.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc_p.update(avg_acc, cnt)
        acces_P_list.append(acc_p.val)

        with torch.no_grad():
            correct = ((domain_logits > 0).float() == domain_label).sum().item()
        acc_num_total += correct
        num += domain_label.size(0)
        acc_d = correct * 1.0 / input.size(0)
        acces_D_list.append(acc_d)
        losses_P_list.append(losses_p.val)
        losses_D_list.append(losses_d.val)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Accuracy_d {3} ({4})\t' \
                  'Loss_d_2 {5}\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_p {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss_mse {mse.val:.5f} ({mse.avg:.5f})\t' \
                  'Adv {adv.val:.6f} ({adv.avg:.6f})\t' \
                  'Loss_d {loss_d.val:.5f} ({loss_d.avg:.5f})\t' \
                  'Accuracy_p {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), acc_d, acc_num_total * 1.0 / num,
                      losses_d_2.val, batch_time=batch_time, data_time=data_time,
                      loss=losses_p, mse=losses_mse, adv=losses_adv, loss_d=losses_d,
                      acc=acc_p)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_P', losses_p.val, global_steps)
            writer.add_scalar('train_loss_mse', losses_mse.val, global_steps)
            writer.add_scalar('train_adv', losses_adv.val, global_steps)
            writer.add_scalar('train_acc_P', acc_p.val, global_steps)
            writer.add_scalar('train_loss_D', losses_d.val, global_steps)
            writer.add_scalar('train_loss_D_2', losses_d_2.val, global_steps)
            writer.add_scalar('train_acc_D', acc_d, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred * 4, outputs, prefix)
    return losses_P_list, losses_D_list, acces_P_list, acces_D_list
