"""Mueve los gaussianos del robot según la pose de PyBullet, vectorizado.

Hace lo mismo que get_transfomration_list + get_segmented_indices + transform_means de
splatsim/utils/robot_splat_render_utils.py (que siguen ahí para los servidores heredados), pero:
  - lo que no depende de la pose (la segmentación, el paso al marco global y la vuelta al del splat
    de los gaussianos que no se mueven) se calcula una sola vez;
  - las rotaciones de todos los links se aplican en una sola operación, y las matrices de Wigner
    de los armónicos esféricos de todos los links en una sola llamada, con una sola copia CPU<->GPU.
La versión por link hacía cientos de operaciones pequeñas y ~60 sincronizaciones con la GPU en cada
observación: era ~2/3 del tiempo de cada ensayo (~0.23 s por observación).
"""
import numpy as np
import pybullet as p
import torch
from e3nn import o3

from splatsim.utils.utils_fk import compute_transformation

# yzx -> xyz, como en transform_shs
_P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)


class RobotSplatTransform:
    """Precalcula lo fijo de un splat de escena con robot; __call__ da los gaussianos movidos.

    pc: el GaussianModel sin deformar (no debe cambiar después). labels: link de cada gaussiano
    dentro de la caja `aabb` del robot (en el orden de esos gaussianos). trans, trans_scale,
    inv_trans: transformations_cache[robot] (PyBullet -> splat)."""

    def __init__(self, robot_uid, pc, labels, trans, trans_scale, inv_trans, aabb):
        self.num_joints = p.getNumJoints(robot_uid)
        with torch.no_grad():
            means = pc.get_xyz
            R, t = trans[:3, :3], trans[:3, 3]
            self.inv_R, self.inv_t = inv_trans[:3, :3], inv_trans[:3, 3]
            # mismo cálculo que get_segmented_indices / transform_means, una sola vez
            self.xyz_global = torch.matmul(R, means.T).T + t
            inside = ((self.xyz_global[:, 0] > aabb[0][0]) & (self.xyz_global[:, 0] < aabb[1][0])
                      & (self.xyz_global[:, 1] > aabb[0][1]) & (self.xyz_global[:, 1] < aabb[1][1])
                      & (self.xyz_global[:, 2] > aabb[0][2]) & (self.xyz_global[:, 2] < aabb[1][2]))
            inside = torch.where(inside)[0]
            labels = labels.long()
            keep = (labels >= 0) & (labels < self.num_joints)
            idx, lab = inside[keep], labels[keep]
            order = torch.argsort(lab, stable=True)
            self.idx, self.lab = idx[order], lab[order]       # gaussianos que se mueven, por link
            counts = torch.bincount(self.lab, minlength=self.num_joints).tolist()
            bounds = np.cumsum([0, *counts])
            self.slices = [(j, int(bounds[j]), int(bounds[j + 1])) for j in range(self.num_joints)
                           if counts[j] > 0]
            # los que no se mueven igual van al marco global y vuelven (lo mismo que hacía transform_means)
            self.xyz_rest = torch.matmul(self.inv_R, self.xyz_global.T).T + self.inv_t
            self.rotation_matrix = R / trans_scale
            self.scale = trans_scale
            self.rot = pc.get_rotation.clone()
            self.opacity = pc.get_opacity_raw
            self.dc = pc._features_dc.clone()
            self.rest = pc._features_rest.clone()
            self.rot_moving = o3.quaternion_to_matrix(self.rot[self.idx])          # (M, 3, 3)
            self.sh_moving = self.rest[self.idx][:, :15, :].transpose(1, 2)        # (M, 3, 15)

    def link_transforms(self, initial_link_states, link_states) -> tuple[torch.Tensor, torch.Tensor]:
        """(J, 3, 3) y (J, 3): de la pose inicial de cada link a la actual (compute_transformation)."""
        rs, ts = [], []
        for a, b in zip(initial_link_states, link_states):
            r, t = compute_transformation((*a["pos"], np.array(a["q"])), (*b["pos"], np.array(b["q"])))
            rs.append(r)
            ts.append(t)
        device = self.xyz_global.device
        return (torch.from_numpy(np.stack(rs)).to(device).float(),
                torch.from_numpy(np.stack(ts)).to(device).float())

    def __call__(self, initial_link_states, link_states):
        """xyz, rot, opacity, shs_featrest, shs_dc como transform_means, con los links movidos de
        initial_link_states a link_states (ambos de get_curr_link_states). initial_link_states se
        pasa en cada llamada porque el simulador lo actualiza (gripper.py)."""
        if not len(initial_link_states) == len(link_states) == self.num_joints:
            raise ValueError(f"{len(initial_link_states)}/{len(link_states)} links, el robot tiene {self.num_joints}")
        with torch.no_grad():
            r_rel, t = self.link_transforms(initial_link_states, link_states)
            # rotación de cada link en el marco del splat (rot_rotation_matrix de transform_means)
            M = (self.inv_R * self.scale) @ r_rel @ self.rotation_matrix          # (J, 3, 3)
            lab = self.lab
            moved = torch.bmm(r_rel[lab], self.xyz_global[self.idx].unsqueeze(-1)).squeeze(-1) + t[lab]
            xyz = self.xyz_rest.clone()
            xyz[self.idx] = torch.matmul(self.inv_R, moved.T).T + self.inv_t
            rot = self.rot.clone()
            rot[self.idx] = o3.matrix_to_quaternion(M[lab] @ self.rot_moving)

            # armónicos esféricos: una matriz de Wigner por link (bandas l = 1..3), como transform_shs
            P = _P.to(M.device)
            angles = o3._rotation.matrix_to_angles(torch.linalg.inv(P) @ M @ P)
            a, b, c = (x.cpu() for x in angles)
            D = torch.zeros(self.num_joints, 15, 15)
            D[:, 0:3, 0:3] = o3.wigner_D(1, a, -b, c)
            D[:, 3:8, 3:8] = o3.wigner_D(2, a, -b, c)
            D[:, 8:15, 8:15] = o3.wigner_D(3, a, -b, c)
            D = D.to(device=M.device, dtype=self.sh_moving.dtype)
            sh = torch.empty_like(self.sh_moving)
            for j, s, e in self.slices:
                sh[s:e] = torch.einsum("ij,nrj->nri", D[j], self.sh_moving[s:e])
            rest = self.rest.clone()
            moving_rest = rest[self.idx]
            moving_rest[:, :15, :] = sh.transpose(1, 2)
            rest[self.idx] = moving_rest
        return xyz, rot, self.opacity, rest, self.dc
