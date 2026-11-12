"""SMIL body model in plain numpy (no chumpy at render time).

Loads smil_web.pkl (the pickle itself needs chumpy installed, because `shapedirs` is a
chumpy object) and evaluates the same model as smil_webuser/verts.py:
shape blend shapes + pose blend shapes ('lrotmin') + linear blend skinning.
Verified against smil_webuser.serialization.load_model by tools in generate_synthetic.py
(`--selftest`).
"""
import pickle

import numpy as np


def rodrigues(r):
    """(N,3) axis-angle -> (N,3,3) rotation matrices."""
    r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(r, axis=1, keepdims=True)
    k = r / np.maximum(theta, 1e-12)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], 1).reshape(-1, 3, 3)
    s = np.sin(theta)[:, :, None]
    c = np.cos(theta)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


class SMIL(object):
    # COCO-17 order: nose, l_eye, r_eye, l_ear, r_ear, l_sho, r_sho, l_elb, r_elb,
    # l_wri, r_wri, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank.
    # Face points are SMPL-topology vertex ids (smplifyx/vertex_ids.py, 'smplh');
    # body points are SMIL joints (MINI-RGBD jointlist.txt).
    COCO_VERTS = {0: 332, 1: 2800, 2: 6260, 3: 583, 4: 4071}
    COCO_JOINTS = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
                   11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}

    def __init__(self, pkl_path):
        d = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
        if d.get('bs_type') != 'lrotmin' or d.get('bs_style') != 'lbs':
            raise ValueError('unexpected SMIL model type %s/%s' % (d.get('bs_type'), d.get('bs_style')))
        self.v_template = np.asarray(d['v_template'], np.float64)
        sd = d['shapedirs']
        self.shapedirs = np.asarray(sd.r if hasattr(sd, 'r') else sd, np.float64)
        self.posedirs = np.asarray(d['posedirs'], np.float64)
        self.J_regressor = d['J_regressor'].toarray() if hasattr(d['J_regressor'], 'toarray') \
            else np.asarray(d['J_regressor'])
        self.weights = np.asarray(d['weights'], np.float64)
        self.f = np.asarray(d['f'], np.int64)
        kt = np.asarray(d['kintree_table'])
        id_to_col = {kt[1, i]: i for i in range(kt.shape[1])}
        self.parent = np.array([-1] + [id_to_col[kt[0, i]] for i in range(1, kt.shape[1])])
        self.n_betas = self.shapedirs.shape[2]

    def __call__(self, pose, betas):
        """pose (72,) axis-angle, betas (<=n_betas,). Returns verts (6890,3), joints (24,3)."""
        betas = np.zeros(self.n_betas) if betas is None else np.asarray(betas, np.float64).ravel()
        b = np.zeros(self.n_betas)
        b[:min(len(betas), self.n_betas)] = betas[:self.n_betas]
        v_shaped = self.v_template + self.shapedirs.dot(b)
        J = self.J_regressor.dot(v_shaped)
        R = rodrigues(pose)                                    # (24,3,3)
        lrotmin = (R[1:] - np.eye(3)[None]).ravel()
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        G = np.zeros((24, 4, 4))
        for i in range(24):
            T = np.eye(4)
            T[:3, :3] = R[i]
            T[:3, 3] = J[i] - (J[self.parent[i]] if i > 0 else 0)
            G[i] = T if i == 0 else G[self.parent[i]] @ T
        joints = G[:, :3, 3].copy()
        # remove rest-pose joint location (skinning transforms act on rest-pose vertices)
        Gs = G.copy()
        Gs[:, :3, 3] -= np.einsum('nij,nj->ni', G[:, :3, :3], J)
        T = np.einsum('vj,jab->vab', self.weights, Gs)
        verts = np.einsum('vab,vb->va', T[:, :3, :3], v_posed) + T[:, :3, 3]
        return verts, joints

    def coco17(self, verts, joints):
        """(17,3) 3D points in COCO keypoint order."""
        out = np.zeros((17, 3))
        for k, vi in self.COCO_VERTS.items():
            out[k] = verts[vi]
        for k, ji in self.COCO_JOINTS.items():
            out[k] = joints[ji]
        return out
