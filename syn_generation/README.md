## Syntheitc Infant Data Generation

## Table of Contents
  * [Description](#description) 
  * [Environment](#environment)
  * [Quick Start](#quick-start)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)


## Description

This repository contains the fitting code borrowing from [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de/) 
so that fit the Skinned Multi-Infant Linear model (SMIL) model to provided poses, and rendering code to generate 2D synthetic images. 

## Environment
The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA TITAN Xp GPU card. Other platforms or GPU cards are not fully tested.

## Quick Start
### Directory of synthetic infant image generation. 
   ```
   ${syn_generation}
   ├── cfg_files
   ├── data
   ├── models
   ├── output
   ├── priors
   ├── results
   ├── smplifyx
   ├── vposer
   ├── load_prior.py
   ├── README.md
   ├── requirements.txt
   ├── optional-requirements.txt
   └── render
   	├── bg_img
   	├── bodies
   	├── outputs
   	├── textures
   	├── smil_webuser
   	├── image_generation.py
   	├── smil_web.pkl
   	└── template.obj

   ```
### Installation
1. Install [Opendr 0.78](https://pypi.org/project/opendr/)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r optional-requirements.txt
   ```
### Preparation for fitting SMIL
   (1) Download [SMIL](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html) model `smil_web.pkl` and put it in `syn_generation/models` folder, place `smil_pose_prior` file into `syn_generation/priors`.
   (2) Place your infant images and corresponding poses in `syn_generation/data/images` and `syn_generation/data/keypoints` separately, as the example files in these folders.

Note: By downloading and/or using SMIL model, you need to agree to the license terms.
### Fitting SMIL
```
cd syn_generation
python smplifyx/main.py \
   --config cfg_files/fit_smil.yaml \
   --data_folder data \
   --output_folder output \
   --visualize=True \
   --model_folder models

```

### Preparation for rendering
   (1) Download background images from LSUN dataset using [this code](https://github.com/fyu/lsun). Or you can use any other images.
   (2) The infant textures in `syn_generation/render/textures` folder are downloaded from the Moving INfants In RGB-D [MINI-RGBD](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html) dataset. 
   In order to make appearances diverse, you can download adult clothing images from [SMPL for MAYA](https://smpl.is.tue.mpg.de/). You need to agree on SMPL license terms and have access to download.
   (3) Copy SMIL model `smil_web.pkl` to `syn_generation/render` folder.
  
### Rendering
```
cd render
python image_generation.py 

```
The generated synthetic images are saved in `syn_generation/render/output` folder.

## Citation
If you use our code or models in your research, please cite with:

```
@article{huang2020invariant,
  title={Invariant representation learning for infant pose estimation with small data},
  author={Huang, Xiaofei and Fu, Nihang and Liu, Shuangjun and Vyas, Kathan and Farnoosh, Amirreza and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2010.06100},
  year={2020}
}
```

## Acknowledgement
Thanks for the open-source
* SmplifyX: [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image, Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.](https://github.com/vchoutas/smplify-x).
* SURREAL: [Learning from Synthetic Humans (SURREAL), Gül Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev and Cordelia Schmid](https://github.com/gulvarol/surreal)

## Setup and generation in this fork (2026-11-03)
The original renderer (`render/image_generation.py`) needs opendr 0.78, which does not build on
Python >= 3.9 or current macOS, has hard-coded paths, and writes images without labels.
`render/generate_synthetic.py` replaces it: same SMIL model, UV template, textures and pinhole
camera, rendered in numpy/OpenCV, and it writes COCO-17 keypoint annotations in the SyRIP format.

1. Environment: `envs/fidip_syn_macos.yml` (see the comments there for the chumpy step).
2. Assets (licensed, git-ignored, never commit):
   - `render/smil_web.pkl` and `models/smil_web.pkl` (SMIL), `priors/smil_pose_prior.pkl`
   - `render/MINI-RGBD_web/` (MINI-RGBD: 12 sequences x 1000 frames of SMIL pose + betas)
   - `render/textures/infant_txt/` (12 MINI-RGBD infant textures, already in the repo)
3. Backgrounds (LSUN scene validation sets, 10 x 300 images, ~50 MB; the train sets are 2-46 GB each):
   ```
   cd render && mkdir -p bg_img/lsun_lmdb && cd bg_img/lsun_lmdb
   for c in bedroom bridge church_outdoor classroom conference_room dining_room kitchen living_room restaurant tower; do
     curl -C - -o ${c}_val_lmdb.zip http://dl.yf.io/lsun/scenes/${c}_val_lmdb.zip; unzip -q ${c}_val_lmdb.zip
     python ../../../../lsun/data.py export ${c}_val_lmdb --out_dir ../lsun/$c --flat
   done
   ```
4. Generate (ids must not collide with SyRIP: real 1-500, synthetic 10000-10999; file names are
   `train%05d.jpg`, so ids < 100000):
   ```
   cd render
   python generate_synthetic.py --source mini --stride 10 --views 10 --out outputs/<run> --start-id 40000 [--bg-cats bedroom,living_room] [--overlays 50]
   python generate_synthetic.py --source fitted --bodies-dir ../output/results --views 10 --out outputs/<run> --start-id 60000
   ```
   Output: `outputs/<run>/images/`, `annotations.json`, `meta.jsonl` (per-image pose source,
   view angles, texture, background), `args.json`, optional `overlays/`.
5. Build a training data root (symlinks + merged json) and train with `DATASET.ROOT` pointing at it:
   ```
   python tools/make_syrip_aug.py --base data/syrip_fix --gen syn_generation/render/outputs/<run> [--max-per-gen N] --out data/syrip_aug_<name>
   python tools/check_annotations.py data/syrip_aug_<name>/
   ```
Fitting SMIL to new real images (`smplifyx/main.py`) additionally needs the SMPLify GMM pose prior
`priors/gmm_08.pkl` (MPI SMPLify download, licence required), which is not included.

### Procedural clothing textures (2026-11-04)
SURREAL clothing textures need credentials. As an alternative, `render/make_proc_textures.py` paints garments (onesies, sleepers, shirt + pants/shorts, tank + leggings, bodysuit, or diaper only) onto the MINI-RGBD infant textures in the shared SMIL/SMPL UV layout:
```
cd syn_generation/render
python make_proc_textures.py --n 24 --seed 0 --out textures_proc/proc_v1
python generate_synthetic.py --source mini --stride 100 --views 1 --max-images 120 \
    --tex-dir textures_proc/proc_v1 --out outputs/proc_test_v1 --start-id 60000 --seed 3 --overlays 40
```
Keep procedural textures outside `render/textures/`; that folder is the default `--tex-dir` and is searched recursively.
