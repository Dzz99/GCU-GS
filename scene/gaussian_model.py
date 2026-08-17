#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import sys
import math
from cluster_uf import union_find
import open3d as o3d
from torch.utils import dlpack
from utils.loss_utils import compute_weighted_pca_normals
import torch.nn.functional as F
import time
try:
    from pytorch3d.ops import knn_points
except ImportError:
    print("pytorch3d not installed. Please run: pip install pytorch3d")
    sys.exit(1)

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        self.setup_functions()
        self.use_app = False

        # para for resample
        self.knn_K = 16         #近邻邻居
        self.max_angle_deg = 15 #簇聚类法向夹角阈值
        self.size_th = 0.1     #簇聚类聚类占extent比例阈值
        self.residual_th = 0.9  #簇聚类平民残差分位数阈值
        self.min_cluster_num = 10
        self.cluster_num_interval = 190
        self.inter_ratio_thr = 0.01 #簇边界点判定-邻域非同簇占比阈值
        self.inter_rho = 5.0  #CE2计算中，距离权重加权
        self.inter_edge_chunk_size = 65536 #ce2计算中，分组计算大小
        self.inter_topk_ratio = 0.05 #簇边界点参与插点比例
        self.inter_scale_n_ratio = 0.1 #基元初始化，最短轴与长轴尺寸比例
        self.min_intra_neighbors = 4
        self.max_angle_deg_edge = 5
        # self.tao = 0.0001
        self.density_topk = 0.2
        self.first_densify = True

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    #     self.spatial_lr_scale = spatial_lr_scale
        
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
    #     normals = torch.tensor(np.asarray(pcd.normals)).float().cuda()   # (N, 3)
        
    #     N = fused_point_cloud.shape[0]
        
    #     # 1) 初始化 scales：第一维（最短轴）是其他两维的 1/10
    #     dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))

    #     dist_q80 = torch.quantile(dist, 0.8)
    #     dist_cap = min(0.05 * spatial_lr_scale, float(dist_q80.item()))

    #     # 截断后的 dist
    #     dist_clamped = torch.clamp(dist, max=dist_cap)

    #     base_scale = torch.log(dist_clamped)[..., None].repeat(1, 3)
    #     scales = base_scale.clone()
    #     scales[:, 0] = base_scale[:, 1] - np.log(10.0)   # 第一维 = 其他维的 1/10
        
    #     # # 组装旋转矩阵 (N, 3, 3)，第一列为 normal
    #     # rotation_matrices = torch.stack([e1, e2, e3], dim=2)   # (N, 3, 3)
    #     rotation_matrices = self._build_local_frame(normals)
        
    #     # 3) 转四元数（pytorch3d 要求输入是 (N, 3, 3)）
    #     rots = matrix_to_quaternion(rotation_matrices)         # (N, 4)
        
    #     # 其余初始化保持不变
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])
    #     opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    #     knn_f = torch.randn((N, 6)).float().cuda()
        
    #     # 参数赋值
    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
    #     self.max_radii2D = torch.zeros((N,), device="cuda")
    #     self.max_weight = torch.zeros((N,), device="cuda")
        
    #     print(f"Number of points at initialisation : {N} (with normal-based orientation)")


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # # print(f"new scale {torch.quantile(dist, 0.1)}")
        # dist_q80 = torch.quantile(dist, 0.8)
        # dist_cap = min(0.05 * spatial_lr_scale, float(dist_q80.item()))

        # # 截断后的 dist
        # dist_clamped = torch.clamp(dist, max=dist_cap)

        # base_scale = torch.log(dist_clamped)[..., None].repeat(1, 3)
        # scales = base_scale.clone()

        # scales = torch.log(dist)[...,None].repeat(1, 3)
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.detach().clone().float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.ini_points = fused_point_cloud.requires_grad_(False)
        self.ini_scaling = scales.requires_grad_(False)
        self.ini_opacity = inverse_sigmoid(0.99 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(False)



    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.geom_densify = training_args.geom_densify
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        self.knn_K = training_args.k_search
        self.density_topk = training_args.topk
        self.max_angle_deg = training_args.max_angle_deg
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.05))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "knn_f": new_knn_f,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, size_mask=None, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if size_mask != None:
            selected_pts_mask = torch.logical_or(selected_pts_mask, size_mask)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            q = max(0.0, min(1.0, 1.0 - ratio))
            threshold = torch.quantile(padded_grad, q)
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                q = max(0.0, min(1.0, 1.0 - ratio))
                threshold = torch.quantile(padded_grads_abs, q)
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            q = max(0.0, min(1.0, 1.0 - ratio))
            threshold = torch.quantile(grads_tmp, q)
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size, iter):
        if self.first_densify:
            self.ini_rotation = self._rotation.requires_grad_(False)
            self.first_densify = False
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0

        max_radii2D = self.max_radii2D.clone()
        # if iter > 2000:
        #     max_grad = 0.0003
        #     max_grad = max_grad / 10

        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        q = max(0.0, min(1.0, 1.0 - ratio))
        q = 1
        Q = torch.quantile(grads_abs.reshape(-1), q)

        size_mask = None
        self.densify_and_clone(grads, max_grad, extent)

        # if iter <= 2000:
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     size_mask = torch.logical_or(big_points_vs, big_points_ws)

        self.densify_and_split(grads, max_grad, grads_abs, Q, extent, max_radii2D, size_mask)
        prune_mask = (self.get_opacity < (min_opacity)).squeeze()
        self.prune_points(prune_mask)

        if self.geom_densify and iter > 2000 and iter%200==0:
            self.geom_densify_prepass(extent, iter)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1



    @torch.no_grad()
    def boundary_by_tangent_angle_gap(self,
        xyz: torch.Tensor,          # (N,3)
        normal: torch.Tensor,       # (N,3)
        cluster_id: torch.Tensor,   # (N,)
        knn: torch.Tensor,          # (N,K) global or intra-cluster knn
        gap_thr: float = math.radians(90.0),
        min_valid_k: int = 8,
        eps: float = 1e-8,
    ):
        """
        基于切平面角度缺口提取单点边界性：
        - 仅使用同簇邻居
        - 用点法向构造切平面
        - 邻居方向投影到切平面，计算最大角度缺口
        Returns:
            boundary_mask: (N,) bool
            max_gap:       (N,) float
            valid_count:   (N,) long
        """
        device = xyz.device
        N, K = knn.shape

        # ---- gather neighbors ----
        nbr = xyz[knn]                              # (N,K,3)
        center = xyz[:, None, :]                    # (N,1,3)
        v = nbr - center                            # (N,K,3)

        cid_c = cluster_id[:, None]                 # (N,1)
        cid_nb = cluster_id[knn]                    # (N,K)
        same_mask = (cid_nb == cid_c) & (cid_c != -1)

        # 可选：排除零向量/self
        dist = torch.linalg.norm(v, dim=-1)         # (N,K)
        same_mask = same_mask & (dist > eps)

        # ---- tangent plane projection by normal ----
        n = F.normalize(normal, dim=-1)             # (N,3)

        ref = torch.zeros_like(n)
        ref[:, 0] = 1.0
        bad = n[:, 0].abs() > 0.9
        ref[bad, 0] = 0.0
        ref[bad, 1] = 1.0

        t1 = F.normalize(torch.cross(n, ref, dim=-1), dim=-1)   # (N,3)
        t2 = torch.cross(n, t1, dim=-1)                         # (N,3)

        # project to tangent plane
        u = (v * t1[:, None, :]).sum(dim=-1)       # (N,K)
        w = (v * t2[:, None, :]).sum(dim=-1)       # (N,K)
        tan_norm = torch.sqrt(u * u + w * w)       # (N,K)

        valid = same_mask & (tan_norm > eps)
        valid_count = valid.sum(dim=1)             # (N,)

        # angle in [-pi, pi]
        theta = torch.atan2(w, u)                  # (N,K)

        # invalid -> +inf, so they go to the end after sorting
        inf = torch.tensor(float("inf"), device=device, dtype=theta.dtype)
        theta_masked = torch.where(valid, theta, inf)

        theta_sorted, _ = torch.sort(theta_masked, dim=1)   # (N,K)

        # pairwise gaps inside valid sorted angles
        theta_next = theta_sorted[:, 1:]
        theta_cur  = theta_sorted[:, :-1]
        pair_valid = torch.isfinite(theta_next) & torch.isfinite(theta_cur)

        gaps = torch.where(pair_valid, theta_next - theta_cur, torch.zeros_like(theta_cur))
        max_inner_gap = gaps.max(dim=1).values

        # wrap gap: first_valid + 2pi - last_valid
        finite_mask = torch.isfinite(theta_sorted)
        first_idx = finite_mask.float().argmax(dim=1)  # first True

        # last valid index
        valid_num = finite_mask.sum(dim=1)
        last_idx = torch.clamp(valid_num - 1, min=0)

        row_idx = torch.arange(N, device=device)
        first_theta = theta_sorted[row_idx, first_idx]
        last_theta  = theta_sorted[row_idx, last_idx]

        wrap_gap = torch.where(
            valid_num >= 2,
            first_theta + 2 * math.pi - last_theta,
            torch.zeros_like(first_theta)
        )

        max_gap = torch.maximum(max_inner_gap, wrap_gap)

        boundary_mask = (valid_count >= min_valid_k) & (max_gap > gap_thr)
        return boundary_mask, max_gap, valid_count



    @torch.no_grad()
    def _safe_normalize(self, x: torch.Tensor, eps: float = 1e-12):
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        x_unit = x / x_norm
        return x_unit, x_norm.squeeze(-1)


    @torch.no_grad()
    def compute_meanshift_vector(self,
        xyz: torch.Tensor,                         # (N, 3), float
        knn: torch.Tensor,                         # (N, K), long
        dist2: torch.Tensor,                       # (N, K), float
        neighbor_valid_mask: torch.Tensor = None, # (N, K), bool
        eps: float = 1e-8,
    ):
        """
        3D mean shift：
            m_i = sum_j (x_j - x_i) w_ij / sum_j w_ij
            w_ij = exp( -||x_i-x_j||^2 / (2 h_i^2) )
        """
        device = xyz.device
        dtype = xyz.dtype
        N, K = knn.shape

        nbr = xyz[knn]                           # (N, K, 3)
        ctr = xyz[:, None, :]                    # (N, 1, 3)
        dvec = nbr - ctr                         # (N, K, 3)
        dist = torch.sqrt(dist2)                 # (N, K)

        if neighbor_valid_mask is None:
            neighbor_valid_mask = torch.ones((N, K), dtype=torch.bool, device=device)

        valid_cnt = neighbor_valid_mask.sum(dim=1)  # (N,)

        valid_cnt_safe = valid_cnt.clamp_min(1)
        h_i = (dist * neighbor_valid_mask.to(dtype)).sum(dim=1) / valid_cnt_safe.to(dtype)

        h_i = h_i.clamp_min(eps)                 # (N,)

        w = torch.exp(-dist.square() / (2.0 * h_i[:, None].square() + eps))   # (N, K)
        w = w * neighbor_valid_mask.to(dtype)

        w_sum = w.sum(dim=1, keepdim=True).clamp_min(eps)                     # (N, 1)
        ms_vec = (w[..., None] * dvec).sum(dim=1) / w_sum                     # (N, 3)

        ms_unit, ms_norm = self._safe_normalize(ms_vec, eps=eps)

        return ms_unit, ms_norm, h_i, valid_cnt

    @torch.no_grad()
    def grouped_topk_mask(self,
        score: torch.Tensor,         # (N,)
        cluster_id: torch.Tensor,    # (N,)
        valid_mask: torch.Tensor,    # (N,) bool
    ):
        """
        return:
            selected_mask: (N,) bool
            selected_idx : (M,) long
        """
        device = score.device
        N = score.shape[0]

        idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return torch.zeros(N, dtype=torch.bool, device=device)

        cid = cluster_id[idx]
        sc = score[idx]

        # 先按 score 降序，再按 cluster_id 升序稳定排序
        order_score = torch.argsort(sc, descending=True, stable=True)
        idx1 = idx[order_score]
        cid1 = cid[order_score]

        order_cluster = torch.argsort(cid1, descending=False, stable=True)
        idx_sorted = idx1[order_cluster]
        cid_sorted = cid1[order_cluster]

        M = idx_sorted.numel()
        new_group = torch.ones(M, dtype=torch.bool, device=device)
        new_group[1:] = cid_sorted[1:] != cid_sorted[:-1]

        group_start = torch.nonzero(new_group, as_tuple=False).squeeze(1)
        group_end = torch.cat(
            [group_start[1:], torch.tensor([M], device=device, dtype=group_start.dtype)],
            dim=0
        )
        group_len = group_end - group_start

        repeated_group_start = torch.repeat_interleave(group_start, group_len)

        fraction = min(
                    float(self.density_topk),
                    float(self.topk_points) / float(max(idx.numel(), 1)),
                    1.0
                    )
        # fraction = self.density_topk
        group_threshold = torch.clamp((group_len.float() * fraction).floor().long(), min=1)
        repeated_threshold = torch.repeat_interleave(group_threshold, group_len)
        rank_in_group = torch.arange(M, device=device) - repeated_group_start
        keep = rank_in_group < repeated_threshold
        selected_idx = idx_sorted[keep]

        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
        selected_mask[selected_idx] = True
        return selected_mask


    @torch.no_grad()
    def project_dir_to_tangent_plane(self,
        direction: torch.Tensor,   # (N, 3)
        normal: torch.Tensor,      # (N, 3)
        eps: float = 1e-12,
    ):
        # 投影到切平面
        dot = (direction * normal).sum(dim=-1, keepdim=True)   # (N,1)
        dir_tan = direction - dot * normal                     # (N,3)

        # 归一化
        dir_tan_norm = torch.linalg.norm(dir_tan, dim=-1, keepdim=True)  # (N,1)
        valid_mask = (dir_tan_norm.squeeze(-1) > eps)

        dir_tan_unit = dir_tan / dir_tan_norm.clamp_min(eps)

        return dir_tan_unit, valid_mask


    @torch.no_grad()
    def sample_points_by_meanshift(self,
        xyz: torch.Tensor,                 # (N, 3) float cuda
        knn: torch.Tensor,                 # (N, K) long  cuda
        dist2: torch.Tensor,               # (N, 3) float cuda
        normal: torch.Tensor,              # (N, 3) float cuda
        cluster_id: torch.Tensor,          # (N,)   long  cuda, -1 表示无效/噪声
        eps: float = 1e-8,
        mask_points = None
    ):

        # ---------- 1) 全局 mean shift ----------
        ms_g_u, ms_g_n, h_g, cnt_g = self.compute_meanshift_vector(
            xyz=xyz,
            knn=knn,
            dist2=dist2,
            neighbor_valid_mask=None,
            eps=eps,
        )

        # ---------- 2) 同簇 mean shift ----------
        nbr_cid = cluster_id[knn]          # (N, K)
        ctr_cid = cluster_id[:, None]      # (N, 1)

        intra_mask = (ctr_cid != -1) & (nbr_cid != -1) & (nbr_cid == ctr_cid)

        ms_i_u, ms_i_n, h_i, cnt_i = self.compute_meanshift_vector(
            xyz=xyz,
            knn=knn,
            dist2=dist2,
            neighbor_valid_mask=intra_mask,
            eps=eps,
        )

        # ---------- 3) 方向一致性 ----------
        consistency_cos = (ms_g_u * ms_i_u).sum(dim=-1)   # (N,)
        cos_th = math.cos(math.radians(self.max_angle_deg_edge))
        # ---------- 4) 基础 mask ----------
        # base_mask = (
        #     (cluster_id != -1) &
        #     (cnt_i >= self.min_intra_neighbors) &
        #     (consistency_cos >= cos_th) &
        #     (ms_i_n >= self.tao)
        # )
        base_mask = (
            (cluster_id != -1) &
            (cnt_i >= self.min_intra_neighbors) &
            (consistency_cos >= cos_th)
        )

        # ---------- 5) 分簇 top-k ----------
        # 采用同簇 mean shift 幅值作为簇内排序分数

        mask_topk = self.grouped_topk_mask(
            score=ms_i_n,
            cluster_id=cluster_id,
            valid_mask=base_mask
        )

        # ---------- 6) 反方向采样 ----------
        # mean shift 指向更高密度方向，因此反方向是更稀疏方向
        sample_dir = -ms_i_u                               # (N, 3)
        sample_dir_tan, tan_valid_mask = self.project_dir_to_tangent_plane(
            direction=sample_dir,
            normal=normal,     # (N, 3)
        )

        final_mask = tan_valid_mask & mask_topk
        idx_selected = mask_points[final_mask]

        cov6 = (self.get_covariance())[idx_selected]
        cov = torch.zeros((cov6.shape[0], 3, 3), dtype=cov6.dtype, device=cov6.device)
        cov[:, 0, 0] = cov6[:, 0]
        cov[:, 0, 1] = cov6[:, 1]
        cov[:, 1, 0] = cov6[:, 1]
        cov[:, 0, 2] = cov6[:, 2]
        cov[:, 2, 0] = cov6[:, 2]
        cov[:, 1, 1] = cov6[:, 3]
        cov[:, 1, 2] = cov6[:, 4]
        cov[:, 2, 1] = cov6[:, 4]
        cov[:, 2, 2] = cov6[:, 5]
        dirs = sample_dir_tan[final_mask]
        step = torch.sqrt(torch.clamp_min(torch.einsum('bi,bij,bj->b', dirs, cov, dirs),1e-12))
        new_xyz = xyz[final_mask] + 0.8 * step.unsqueeze(-1) * dirs
        new_scaling = self.scaling_activation(self._scaling[idx_selected])
        new_rotation = (self.get_rotation_matrix())[idx_selected]

        new_opacity = self._opacity[idx_selected]
        return new_xyz, new_opacity, new_scaling, new_rotation


    # @torch.no_grad()
    # def geom_densify_prepass(self, extent, iter):
    #     """
    #     This function:
    #       1) gets normals,
    #       2) builds KNN,
    #       3) clusters via union-find,
    #       4) inter-cluster insertion,
    #       5) intra-cluster insertion,
    #       6) merges + unique/cap,
    #       7) returns densifymask for split and clone.
    #     """
    #     self.cluster_num=int(self.min_cluster_num+self.cluster_num_interval*(iter-500)/(9000-1000))
    #     self.density_topk = self.density_topk-(self.density_topk - 0.01)*(iter-500)/(9000-1000)
    #     # self.inter_topk_ratio = self.inter_topk_ratio-(self.inter_topk_ratio - 0.01)*(iter-500)/(8000-1000)
    #     ini_num = self._xyz.shape[0]
    #     add_num = self.ini_points.shape[0]
    #     feat_dc_tail = self._features_dc.shape[1:]        # e.g. (Cdc,) or (3,1) etc.
    #     feat_rest_tail = self._features_rest.shape[1:]    # e.g. (Crest,) or (...)
    #     new_features_dc = torch.zeros((add_num, *feat_dc_tail), device="cuda", dtype=self._features_dc.dtype)
    #     new_features_rest = torch.zeros((add_num, *feat_rest_tail), device="cuda", dtype=self._features_rest.dtype)
    #     new_knn_f = torch.randn((add_num, 6)).float().cuda()
    #     self.densification_postfix(self.ini_points, new_knn_f, new_features_dc, new_features_rest, self.ini_opacity, self.ini_scaling, self.ini_rotation)


    #     # ---------- 1) normals ----------
    #     x = self._xyz.detach().contiguous()
    #     N = x.shape[0]

    #     # ---------- 2) build KNN ----------
    #     k_search = min(self.knn_K + 1, N)

    #     try:
    #         from pytorch3d.ops import knn_points
    #     except ImportError:
    #         print("pytorch3d not installed. Please run: pip install pytorch3d")
    #         sys.exit(1)

    #     # knn_points 需要 batch 维度 (B, N, D)，这里 B=1
    #     knn_res = knn_points(
    #         x.unsqueeze(0),          # query (1, N, 3)
    #         x.unsqueeze(0),          # reference (1, N, 3)
    #         K=k_search,          # 包含自身
    #         return_sorted=True
    #     )
    #     dist2, knn = knn_res.dists.squeeze(0)[:, 1:k_search], knn_res.idx.squeeze(0)[:, 1:k_search]   # (N, K+1)

    #     # Ensure dtypes
    #     knn = knn.to(torch.int32)
    #     dist2 = dist2.to(torch.float32)
    #     _, normal = compute_weighted_pca_normals(x, self.get_opacity, knn, dist2, radius=0.1*extent)
    #     # normal = self.get_smallest_axis()
    #     # normal = F.normalize(normal, dim=-1, eps=1e-6)

    #     # ---------- 3) residual ----------
    #     residual = self._compute_plane_residual(normal, knn, dist2)

    #     # ---------- 4) clustering ----------
    #     # returns cluster_id (N,)
    #     cluster_id= self._cluster_union_find(normal=normal, knn=knn, dist2=dist2, residual=residual, extent=extent)
    #     prune_mask = (cluster_id == -1)
    #     prune_mask = prune_mask[:self._xyz.shape[0]]
    #     op_cluster = self.get_opacity[prune_mask] / 10.0
    #     op_cluster = torch.clamp(op_cluster, min=0.01)
    #     self._opacity[prune_mask] = self.inverse_opacity_activation(op_cluster)

    #     # op_cluster = self.inverse_opacity_activation((self.get_opacity[prune_mask]) / 10)
    #     # self._opacity[prune_mask] = 
    #     # def save_cluster_colored_ply(
    #     #     xyz,
    #     #     cluster_id,
    #     #     save_path,
    #     #     boundary_mask=None,
    #     #     boundary_color=(255, 0, 0),   # 边界点高亮颜色：默认红色
    #     #     invalid_color=(0, 0, 0),      # cluster_id < 0 的颜色
    #     # ):
    #     #     """
    #     #     xyz:           [N, 3] torch.Tensor / np.ndarray
    #     #     cluster_id:    [N]    torch.Tensor / np.ndarray
    #     #     boundary_mask: [N]    bool Tensor / np.ndarray / None
    #     #                 True 的点会被 boundary_color 高亮覆盖
    #     #     save_path:     str
    #     #     """

    #     #     # ---- 转到 CPU / numpy ----
    #     #     if isinstance(xyz, torch.Tensor):
    #     #         xyz = xyz.detach().cpu().numpy()
    #     #     if isinstance(cluster_id, torch.Tensor):
    #     #         cluster_id = cluster_id.detach().cpu().numpy()
    #     #     if boundary_mask is not None and isinstance(boundary_mask, torch.Tensor):
    #     #         boundary_mask = boundary_mask.detach().cpu().numpy()

    #     #     xyz = np.asarray(xyz, dtype=np.float32)
    #     #     cluster_id = np.asarray(cluster_id).reshape(-1)

    #     #     if boundary_mask is not None:
    #     #         boundary_mask = np.asarray(boundary_mask).reshape(-1).astype(bool)

    #     #     assert xyz.shape[0] == cluster_id.shape[0]
    #     #     assert xyz.shape[1] == 3
    #     #     if boundary_mask is not None:
    #     #         assert boundary_mask.shape[0] == xyz.shape[0]

    #     #     N = xyz.shape[0]

    #     #     # ---- 为每个 cluster 分配颜色 ----
    #     #     colors = np.zeros((N, 3), dtype=np.uint8)

    #     #     unique_ids = np.unique(cluster_id)

    #     #     rng = np.random.default_rng(42)  # 固定随机种子，保证每次颜色一致
    #     #     color_map = {}

    #     #     invalid_color = np.array(invalid_color, dtype=np.uint8)
    #     #     boundary_color = np.array(boundary_color, dtype=np.uint8)

    #     #     for cid in unique_ids:
    #     #         if cid < 0:
    #     #             color_map[cid] = invalid_color
    #     #         else:
    #     #             color_map[cid] = rng.integers(0, 256, size=3, dtype=np.uint8)

    #     #     for cid in unique_ids:
    #     #         mask = (cluster_id == cid)
    #     #         colors[mask] = color_map[cid]

    #     #     # ---- 边界点高亮覆盖 ----
    #     #     if boundary_mask is not None:
    #     #         colors[boundary_mask] = boundary_color

    #     #     # ---- 写 ASCII PLY ----
    #     #     with open(save_path, "w") as f:
    #     #         f.write("ply\n")
    #     #         f.write("format ascii 1.0\n")
    #     #         f.write(f"element vertex {N}\n")
    #     #         f.write("property float x\n")
    #     #         f.write("property float y\n")
    #     #         f.write("property float z\n")
    #     #         f.write("property uchar red\n")
    #     #         f.write("property uchar green\n")
    #     #         f.write("property uchar blue\n")
    #     #         f.write("end_header\n")

    #     #         for i in range(N):
    #     #             x, y, z = xyz[i]
    #     #             r, g, b = colors[i]
    #     #             f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


    #     # def save_normal_colored_ply(xyz, normal, save_path):
    #     #     """
    #     #     保存 PLY 文件，同时包含：
    #     #     - 坐标 (x, y, z)
    #     #     - 法向量 (nx, ny, nz) —— 用于 CloudCompare 查看法向量方向
    #     #     - 伪彩色 (red, green, blue) —— 基于法向量映射到 [-1,1] → [0,255]

    #     #     参数:
    #     #         xyz:    [N, 3] torch.Tensor / np.ndarray
    #     #         normal: [N, 3] torch.Tensor / np.ndarray（已归一化或未归一化均可）
    #     #         save_path: str
    #     #     """
    #     #     # 统一转为 numpy float32
    #     #     if isinstance(xyz, torch.Tensor):
    #     #         xyz = xyz.detach().cpu().numpy()
    #     #     if isinstance(normal, torch.Tensor):
    #     #         normal = normal.detach().cpu().numpy()
            
    #     #     xyz = np.asarray(xyz, dtype=np.float32)
    #     #     normal = np.asarray(normal, dtype=np.float32)
            
    #     #     assert xyz.shape == normal.shape, "xyz 和 normal 形状必须一致"
    #     #     assert xyz.shape[1] == 3, "必须是 (N, 3) 形状"

    #     #     # 确保 normal 是单位向量（CloudCompare 通常期望单位法向量）
    #     #     norm = np.linalg.norm(normal, axis=1, keepdims=True)
    #     #     norm = np.clip(norm, 1e-8, None)
    #     #     normal_unit = normal / norm

    #     #     # 法向量映射到伪彩色（保持你原来的颜色逻辑）
    #     #     colors = 0.5 * (normal_unit + 1.0)
    #     #     colors = np.clip(colors, 0.0, 1.0)
    #     #     colors = (colors * 255).astype(np.uint8)

    #     #     N = xyz.shape[0]

    #     #     # 以 ASCII 格式写入（CloudCompare 完全支持）
    #     #     with open(save_path, "w") as f:
    #     #         f.write("ply\n")
    #     #         f.write("format ascii 1.0\n")
    #     #         f.write(f"element vertex {N}\n")
    #     #         f.write("property float x\n")
    #     #         f.write("property float y\n")
    #     #         f.write("property float z\n")
    #     #         f.write("property float nx\n")     # 新增：法向量 x
    #     #         f.write("property float ny\n")     # 新增：法向量 y
    #     #         f.write("property float nz\n")     # 新增：法向量 z
    #     #         f.write("property uchar red\n")
    #     #         f.write("property uchar green\n")
    #     #         f.write("property uchar blue\n")
    #     #         f.write("end_header\n")
                
    #     #         for i in range(N):
    #     #             x, y, z = xyz[i]
    #     #             nx, ny, nz = normal_unit[i]      # 保存单位法向量
    #     #             r, g, b = colors[i]
    #     #             f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")
        
    #     # save_cluster_colored_ply(self._xyz, cluster_id, "cluster_vis.ply")

    #     # save_normal_colored_ply(self._xyz, normal, "normal_vis.ply")
    #     # sys.exit(0)

    #     # ---------- 5) intra-cluster insertion ----------
    #     intar_pts, opacity_intar, intar_s, intar_R = self.sample_points_by_meanshift(
    #         xyz = x, knn=knn, dist2=dist2, normal = normal, cluster_id=cluster_id
    #     )

    #     # ---------- 6) inter-cluster insertion ----------
    #     if iter % 500 ==0:
    #         boundary_mask, _, _  = self.boundary_by_tangent_angle_gap(x, normal, cluster_id, knn)
    #         boundary_idx = torch.where(boundary_mask)[0]
    #         # save_cluster_colored_ply(self._xyz, cluster_id, "boundary_vis.ply", boundary_mask)
    #         inter_sel_i, inter_sel_j, xyz_inter, opacity_inter = self._insert_inter_cluster(
    #             xyz = x, normal=normal, knn=knn,
    #             residual=residual, cluster_id=cluster_id, boundary_idx = boundary_idx
    #         )
    #     else:
    #         inter_sel_i = x.new_empty((0,))
    #         inter_sel_j = x.new_empty((0,))
    #         xyz_inter = x.new_empty((0,3))
    #         opacity_inter = self._opacity.new_empty((0,))
        
    #     # Stack candidates: (M,3)
    #     xyz_inter = xyz_inter.contiguous()

    #     # # Choose K for new points 
    #     k_new = min(16, N)

    #     knn_res_new = knn_points(
    #         xyz_inter.unsqueeze(0),          # query (1, N, 3)
    #         x.unsqueeze(0),          # reference (1, N, 3)
    #         K=k_new,          # 包含自身
    #         return_sorted=True
    #     )
    #     dist2_new, knn_new = knn_res_new.dists.squeeze(0), knn_res_new.idx.squeeze(0)   # (1, N, K+1)

    #     # Ensure dtypes
    #     knn_new = knn_new.to(torch.int64)
    #     dist2_new = dist2_new.to(torch.float32)

    #     # new para for inter pts
    #     M_inter = xyz_inter.shape[0]
    #     inter_pts, inter_R, inter_s = self._compute_inter_para(inter_sel_i, inter_sel_j, xyz_inter, normal, knn_new[:M_inter], dist2_new[:M_inter])

    #     # ---------- 7) merge new points  ----------
    #     # pts
    #     # print("inter:{} intar:{}".format(inter_pts.shape[0], intar_pts.shape[0]))
    #     new_xyz = torch.cat([inter_pts, intar_pts], dim=0).contiguous()  # (M,3)
    #     M = new_xyz.shape[0]

    #     n_init_points = self.get_xyz.shape[0]
    #     if new_xyz.shape[0] + n_init_points > self.max_all_points:
    #         print("Point cloud overflow in the geom_densify")
    #         return 

    #     # opacities
    #     new_opacity = torch.cat([opacity_inter, opacity_intar], dim=0).contiguous()
    #     op = self.opacity_activation(new_opacity)
    #     op = torch.clamp(op * 0.8, min=torch.full_like(op, 0.05))
    #     # op = torch.full_like(new_opacity, 0.01)
    #     new_opacity = self.inverse_opacity_activation(op)
    #     # new_opacity = self.inverse_opacity_activation((self.opacity_activation(new_opacity) * 0.8))
    #     if new_opacity.ndim == 1:
    #         new_opacity = new_opacity.view(-1, 1)  # (M,1)

    #     # scale
    #     scaling = torch.cat([inter_s, intar_s], dim=0).contiguous()  # (M,3)
    #     new_scaling = self.scaling_inverse_activation(scaling).contiguous()

    #     # rotation: R -> quaternion 
    #     new_R = torch.cat([inter_R, intar_R], dim=0).contiguous()     # (M,3,3)
    #     new_rotation = matrix_to_quaternion(new_R).contiguous()       # (M,4) in (w,x,y,z)
    #     # new_rotation = torch.zeros((new_xyz.shape[0], 4), device="cuda")
    #     # new_rotation[:, 0] = 1

    #     mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
    #     mask[:ini_num] = False
    #     self.prune_points(mask)

    #     new_features_dc = torch.zeros((M, *feat_dc_tail), device=new_xyz.device, dtype=self._features_dc.dtype)
    #     new_features_rest = torch.zeros((M, *feat_rest_tail), device=new_xyz.device, dtype=self._features_rest.dtype)
    #     new_knn_f = torch.randn((new_xyz.shape[0], 6)).float().cuda()
    #     self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    @torch.no_grad()
    def points_upsample(self, extent, iter, mask_points):
        """
        This function:
          1) builds KNN,
          2) gets normals,
          3) clusters via union-find,
          4) inter-cluster insertion,
          5) intra-cluster insertion,
          6) merges + unique/cap,
          7) returns densifymask for split and clone.
        """

        # ---------- 1) normals ----------
        x = (self._xyz)[mask_points].detach().contiguous()
        N = x.shape[0]

        # ---------- 2) build KNN ----------
        k_search = min(self.knn_K + 1, N)

        # knn_points 需要 batch 维度 (B, N, D)，这里 B=1
        knn_res = knn_points(
            x.unsqueeze(0),          # query (1, N, 3)
            x.unsqueeze(0),          # reference (1, N, 3)
            K=k_search,          # 包含自身
            return_sorted=True
        )
        dist2, knn = knn_res.dists.squeeze(0)[:, 1:k_search], knn_res.idx.squeeze(0)[:, 1:k_search]   # (N, K+1)

        # Ensure dtypes
        knn = knn.to(torch.int32)
        dist2 = dist2.to(torch.float32)
        _, normal = compute_weighted_pca_normals(x, self.get_opacity[mask_points], knn, dist2, radius=0.1*extent)

        # ---------- 3) residual ----------
        residual = self._compute_plane_residual(x, normal, knn, dist2)

        # ---------- 4) clustering ----------
        # returns cluster_id (N,)
        cluster_id= self._cluster_union_find(x, normal=normal, knn=knn, dist2=dist2, \
                                             residual=residual, extent=extent, opacity=self.get_opacity[mask_points])

        prune_mask = (cluster_id == -1)
        global_prune_idx = mask_points[prune_mask]
        op_cluster = self.get_opacity[global_prune_idx] / 10.0
        # op_cluster = torch.clamp(op_cluster, min=0.01)
        self._opacity[global_prune_idx] = self.inverse_opacity_activation(op_cluster)

        # ---------- 5) intra-cluster insertion ----------
        intar_pts, opacity_intar, intar_s, intar_R = self.sample_points_by_meanshift(
            xyz = x, knn=knn, dist2=dist2, normal = normal, cluster_id=cluster_id, mask_points=mask_points
        )

        # ---------- 6) inter-cluster insertion ----------
        if iter % 50000 ==0:
            boundary_mask, _, _  = self.boundary_by_tangent_angle_gap(x, normal, cluster_id, knn)
            boundary_idx = torch.where(boundary_mask)[0]
            # save_cluster_colored_ply(self._xyz, cluster_id, "boundary_vis.ply", boundary_mask)
            inter_sel_i, inter_sel_j, xyz_inter, opacity_inter = self._insert_inter_cluster(
                xyz = x, normal=normal, knn=knn, residual=residual, 
                cluster_id=cluster_id, boundary_idx = boundary_idx, mask_points=mask_points
            )
        else:
            inter_sel_i = x.new_empty((0,))
            inter_sel_j = x.new_empty((0,))
            xyz_inter = x.new_empty((0,3))
            opacity_inter = self._opacity.new_empty((0,))
        
        # Stack candidates: (M,3)
        xyz_inter = xyz_inter.contiguous()

        # # Choose K for new points 
        k_new = min(16, N)

        knn_res_new = knn_points(
            xyz_inter.unsqueeze(0),          # query (1, N, 3)
            x.unsqueeze(0),          # reference (1, N, 3)
            K=k_new,          # 包含自身
            return_sorted=True
        )
        dist2_new, knn_new = knn_res_new.dists.squeeze(0), knn_res_new.idx.squeeze(0)   # (1, N, K+1)

        # Ensure dtypes
        knn_new = knn_new.to(torch.int64)
        dist2_new = dist2_new.to(torch.float32)

        # new para for inter pts
        inter_pts, inter_R, inter_s = self._compute_inter_para(x, inter_sel_i, inter_sel_j, 
                                                               xyz_inter, normal, knn_new, dist2_new)

        # ---------- 7) merge new points  ----------
        # pts
        print("inter:{} intar:{} all{}".format(inter_pts.shape[0], intar_pts.shape[0], N))
        new_xyz = torch.cat([inter_pts, intar_pts], dim=0).contiguous()  # (M,3)

        # opacities
        new_opacity = torch.cat([opacity_inter, opacity_intar], dim=0).contiguous()
        op = self.opacity_activation(new_opacity)
        op = torch.clamp(op * 0.8, min=torch.full_like(op, 0.05))
        # op = torch.full_like(new_opacity, 0.01)
        new_opacity = self.inverse_opacity_activation(op)
        # new_opacity = self.inverse_opacity_activation((self.opacity_activation(new_opacity) * 0.8))
        if new_opacity.ndim == 1:
            new_opacity = new_opacity.view(-1, 1)  # (M,1)

        # scale
        scaling = torch.cat([inter_s, intar_s], dim=0).contiguous()  # (M,3)
        new_scaling = self.scaling_inverse_activation(scaling).contiguous()

        # rotation: R -> quaternion 
        new_R = torch.cat([inter_R, intar_R], dim=0).contiguous()     # (M,3,3)
        new_rotation = matrix_to_quaternion(new_R).contiguous()       # (M,4) in (w,x,y,z)
        
        return new_xyz, new_opacity, new_scaling, new_rotation



    @torch.no_grad()
    def geom_densify_prepass(self, extent, iter):
        """
        This function:
          1) gets normals,
          2) builds KNN,
          3) clusters via union-find,
          4) inter-cluster insertion,
          5) intra-cluster insertion,
          6) merges + unique/cap,
          7) returns densifymask for split and clone.
        """
        self.cluster_num=int(self.min_cluster_num+self.cluster_num_interval*(iter-500)/(10000-500))
        # self.density_topk = self.density_topk-(self.density_topk - 0.01)*(iter-500)/(10000-500)
        # self.inter_topk_ratio = self.inter_topk_ratio-(self.inter_topk_ratio - 0.01)*(iter-500)/(8000-1000)

        feat_dc_tail = self._features_dc.shape[1:]        # e.g. (Cdc,) or (3,1) etc.
        feat_rest_tail = self._features_rest.shape[1:]    # e.g. (Crest,) or (...)
        ini_num = self._xyz.shape[0]
        add_num = self.ini_points.shape[0]
        new_features_dc = torch.zeros((add_num, *feat_dc_tail), device="cuda", dtype=self._features_dc.dtype)
        new_features_rest = torch.zeros((add_num, *feat_rest_tail), device="cuda", dtype=self._features_rest.dtype)
        new_knn_f = torch.randn((add_num, 6)).float().cuda()
        self.densification_postfix(self.ini_points, new_knn_f, new_features_dc, new_features_rest, self.ini_opacity, self.ini_scaling, self.ini_rotation)
        if (ini_num+add_num) > 1_500_000:
            centroid = self._xyz.mean(dim=0)
            rel = self._xyz[:, :2] - centroid[:2]          # 只看 xy 平面
            angles = torch.atan2(rel[:, 1], rel[:, 0])     # (-π, π]
            angles = (angles + 2 * torch.pi) % (2 * torch.pi)  # 转为 [0, 2π)

            # 0~60° 随机初始化角度边界偏移
            offset = torch.rand(1, device="cuda").item() * (torch.pi / 3)
            shifted = (angles + offset) % (2 * torch.pi)

            sector_size = 2 * torch.pi / 12
            sector_id = torch.floor(shifted / sector_size).long()  # 0~5

            # ---------- 3. 按 6 个扇区循环处理 ----------
            new_xyz_list = []
            new_opacity_list = []
            new_scaling_list = []
            new_rotation_list = []
            for sid in range(12):
                sector_idx = torch.nonzero(sector_id == sid, as_tuple=False).squeeze(1)
                n_sec = sector_idx.numel()
                if n_sec < 20:
                    continue

                # if n_sec > 1_000_000:
                #     sector_angles = angles[sector_idx]                     # shape: (n_sec,)

                #     # 2. 按极角排序 → 严格“按照极角分”
                #     sorted_local = torch.argsort(sector_angles)            # 局部排序索引
                #     split = n_sec // 2

                #     sub_idx1 = sector_idx[sorted_local[:split]]            # 第一半（极角较小）
                #     sub_idx2 = sector_idx[sorted_local[split:]]            # 第二半（极角较大）

                #     # 3. 分别处理两个子部分（两次调用）
                #     for sub_idx in (sub_idx1, sub_idx2):
                #         # 可选：子部分太小就跳过（几乎不会发生）
                #         if sub_idx.numel() < 20:
                #             continue
                #         # self.topk_points = 
                #         sector_xyz, sector_opacity, sector_scaling, sector_R = self.points_upsample(
                #             extent, iter, sub_idx
                #         )
                #         new_xyz_list.append(sector_xyz)
                #         new_opacity_list.append(sector_opacity)
                #         new_scaling_list.append(sector_scaling)
                #         new_rotation_list.append(sector_R)

                # else:
                    # ==================== 正常小扇区处理（保持原逻辑） ====================
                self.topk_points = max(1, int(n_sec * 0.01))
                sector_xyz, sector_opacity, sector_scaling, sector_R = self.points_upsample(
                    extent, iter, sector_idx
                )
                new_xyz_list.append(sector_xyz)
                new_opacity_list.append(sector_opacity)
                new_scaling_list.append(sector_scaling)
                new_rotation_list.append(sector_R)

                # sector_xyz, sector_opacity, sector_scaling, sector_R = self.points_upsample(extent, iter, sector_idx)
                # new_xyz_list.append(sector_xyz)
                # new_opacity_list.append(sector_opacity)
                # new_scaling_list.append(sector_scaling)
                # new_rotation_list.append(sector_R)
            if len(new_xyz_list) == 0:
                return

            new_xyz = torch.cat(new_xyz_list, dim=0).contiguous()
            new_opacity = torch.cat(new_opacity_list, dim=0).contiguous()
            new_scaling = torch.cat(new_scaling_list, dim=0).contiguous()
            new_rotation = torch.cat(new_rotation_list, dim=0).contiguous()
        else:

            # mask = torch.ones(ini_num + add_num, dtype=torch.bool, device="cuda")
            mask_idx = torch.arange(self._xyz.shape[0], device="cuda")
            self.topk_points = max(1, int((ini_num+add_num) * 0.01))
            new_xyz, new_opacity, new_scaling, new_rotation = self.points_upsample(extent, iter, mask_idx)


        mask_add = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        mask_add[:ini_num] = False
        self.prune_points(mask_add)

        M = new_xyz.shape[0]
        if M + ini_num > self.max_all_points:
            print("Point cloud overflow in the geom_densify")
            return

        new_features_dc = torch.zeros((M, *feat_dc_tail), device=new_xyz.device, dtype=self._features_dc.dtype)
        new_features_rest = torch.zeros((M, *feat_rest_tail), device=new_xyz.device, dtype=self._features_rest.dtype)
        new_knn_f = torch.randn((M, 6)).float().cuda()
        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)


    @torch.no_grad()
    def _compute_plane_residual(self, x, normal, knn, dist2, eps=1e-8):
        """
        Compute plane residual for each point as a weighted mean
        of neighbor-to-plane normal projection distances.

        Args:
            xyz:    (N,3) float tensor, CUDA
            normal: (N,3) float tensor, CUDA, assumed normalized
            knn:    (N,K) int tensor
            dist2:  (N,K) float tensor, squared distances
            eps:    small constant for numerical stability

        Returns:
            residual: (N,) float tensor
        """
        # (N,K,3): neighbor coordinates
        nbr_xyz = x[knn]

        # (N,1,3): center coordinates
        ctr_xyz = x[:, None, :]

        # (N,K,3): neighbor displacement vectors
        diff = nbr_xyz - ctr_xyz

        # (N,K): absolute normal projection distances
        # | n_i^T (p_j - p_i) |
        proj = (diff * normal[:, None, :]).sum(dim=-1).abs()

        # (N,K): distance-based weights
        # Using inverse squared distance (stable, cheap)
        w = 1.0 / (dist2 + eps)

        # (N,): weighted mean
        residual = (w * proj).sum(dim=1) / (w.sum(dim=1) + eps)

        return residual
    
    def _compute_sparse_score(self, dist2):
        """
        Placeholder:
          Output sparse_score (N,)
        """
        raise NotImplementedError

    # def _cluster_union_find(self, normal, knn, dist2, residual, extent):
    #     """
    #     纯 PyTorch 增量 Union-Find（彻底规避 CUDA union_find 的 cudaMalloc OOM）
    #     专门为 N≈500万 优化，5M 点通常 2~4 秒（RTX 4090 / 3090 / A100）
    #     cluster_id 语义与你原来 CUDA 版本 100% 一致
    #     """
    #     device = self._xyz.device
    #     N = self._xyz.shape[0]
    #     K = knn.shape[1]

    #     # 1) 阈值（和你原来完全一样）
    #     cos_th = math.cos(math.radians(self.max_angle_deg))
    #     max_dist2 = (self.size_th * extent) ** 2

    #     # 2) 初始化 parent
    #     parent = torch.arange(N, device=device, dtype=torch.int64)

    #     # 3) 分批增量 union（核心）
    #     batch_size = 500_000                     # ← 5M 点最优值，可再调大到 262144
    #     normal = normal.contiguous()            # 加速 gather

    #     with torch.no_grad():
    #         for start in range(0, N, batch_size):
    #             end = min(start + batch_size, N)

    #             i_batch = torch.arange(start, end, device=device, dtype=torch.long)
    #             i_idx = i_batch.unsqueeze(1).expand(-1, K)   # (B, K)
    #             j_idx = knn[start:end]
    #             d2    = dist2[start:end]

    #             # mask（和你原来一模一样）
    #             mask = (j_idx != i_idx) & (d2 <= max_dist2)
    #             n_i = normal[i_idx]
    #             n_j = normal[j_idx]
    #             dot = (n_i * n_j).sum(dim=-1)
    #             mask = mask & (dot >= cos_th)

    #             if not mask.any():
    #                 continue

    #             ii = i_idx[mask]
    #             jj = j_idx[mask]

    #             # 增量 union：永远大索引指向小索引
    #             pu = parent[ii]
    #             pv = parent[jj]
    #             diff = pu != pv
    #             if diff.any():
    #                 pu = pu[diff]
    #                 pv = pv[diff]
    #                 p_min = torch.minimum(pu, pv)
    #                 p_max = torch.maximum(pu, pv)
    #                 parent[p_max] = p_min

    #     # 4) 路径压缩直到完全收敛（比固定次数更快）
    #     changed = True
    #     while changed:
    #         new_parent = torch.minimum(parent, parent[parent])
    #         changed = (new_parent != parent).any()
    #         parent = new_parent

    #     cluster_id = parent

    #     # 5) 小簇过滤（和你原来完全一样）
    #     thr = self.cluster_num
    #     if thr > 1:
    #         num_clusters = int(cluster_id.max().item()) + 1
    #         cluster_sizes = torch.zeros(num_clusters, device=device, dtype=torch.long)
    #         cluster_sizes.scatter_add_(0, cluster_id, torch.ones_like(cluster_id))

    #         small_mask = cluster_sizes[cluster_id] < thr
    #         cluster_id = cluster_id.clone()
    #         cluster_id[small_mask] = -1

    #     return cluster_id


    @torch.no_grad()
    def _cluster_union_find(self, xyz, normal, knn, dist2, residual, extent, opacity):
        """
        Build edges from KNN then union-find.

        Args:
            xyz:      (N,3) float, on GPU
            normal:   (N,3) float, unit normals
            knn:      (N,K) long, neighbor indices
            dist2:    (N,K) float, squared distances to knn
            residual: (N,)  float, plane residual proxy (bigger => more curved/noisy)

        Returns:
            cluster_id: (N,) long, cluster label = min index in that component
        """
        device = xyz.device
        N = xyz.shape[0]

        # 1) thresholds
        # 角度阈值：允许的最大法向夹角（度）
        cos_th = math.cos(math.radians(self.max_angle_deg))

        # 残差阈值：
        # res_th = torch.quantile(residual, self.residual_th)

        # 距离阈值：
        max_dist2 = (self.size_th * extent) ** 2
        # 3) build directed edges i -> j from KNN
        K = knn.shape[1]
        i_idx = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1).expand(N, K)  # (N,K)

        j_idx = knn.to(torch.long)                                                          # (N,K)

        # 去掉 self-edge
        mask = (j_idx != i_idx)

        # 距离约束
        mask = mask & (dist2 <= max_dist2)

        # 法向一致性约束：dot(n_i, n_j) >= cos_th
        # normal: (N,3) -> gather neighbor normals (N,K,3)
        n_i = normal[i_idx,:]                         # (N,K,3)
        n_j = normal[j_idx,:]                         # (N,K,3)
        dot = (n_i * n_j).sum(dim=-1)        # (N,K)
        mask = mask & (dot >= cos_th)

        # 残差约束
        # r_i = residual[i_idx]                   # (N,K)
        # r_j = residual[j_idx]                   # (N,K)
        # mask = mask & (torch.maximum(r_i, r_j) <= float(res_th))
        # mask = (opacity.squeeze(-1)[knn] > 0.1) & mask
        # -----------------------------
        # 3) finalize edges tensor (E,2)
        # -----------------------------
        ii = i_idx[mask]
        jj = j_idx[mask]
        if ii.numel() == 0:
            # 没有边：每个点独立成簇，ID=自己
            return torch.arange(N, device=device, dtype=torch.long)

        edges = torch.stack([ii, jj], dim=1)  # (E,2)

        # 去重
        sorted_edges = torch.sort(edges, dim=1)[0]  # 每行排序
        key_e = sorted_edges[:, 0] * N + sorted_edges[:, 1]
        key_e = torch.unique(key_e)
        edges = torch.stack([key_e // N, key_e % N], dim=1).to(torch.int32)

        # -----------------------------
        # 4) union-find
        # -----------------------------
        cluster_id = union_find(edges, N=N)

        cluster_id = cluster_id.to(torch.int64)

        thr = self.cluster_num

        if thr > 1:
            # 方法1：用 scatter 构建簇大小表（最快、最稳）
            num_clusters = int(cluster_id.max().item()) + 1
            cluster_sizes = torch.zeros(num_clusters, device=device, dtype=torch.long)
            cluster_sizes.scatter_add_(0, cluster_id, torch.ones_like(cluster_id))
            
            # 小簇的 cluster_id 列表
            small_mask = cluster_sizes[cluster_id] < thr
            cluster_id[small_mask] = -1

        return cluster_id

    @torch.no_grad()
    def _insert_inter_cluster(self,
                            xyz: torch.Tensor,          # (N,3)
                            normal: torch.Tensor,       # (N,3)
                            knn: torch.Tensor,          # (N,K)
                            residual: torch.Tensor,     # (N,)
                            cluster_id: torch.Tensor,   # (N,)
                            boundary_idx: torch.Tensor, # (N,)
                            mask_points: torch.Tensor,  # (N,)
                            ):
        xyz_b = xyz[boundary_idx]
        cluster_id_b = cluster_id[boundary_idx]
        residual_b = residual[boundary_idx]
        

        k_search = 65

        try:
            from pytorch3d.ops import knn_points
        except ImportError:
            print("pytorch3d not installed. Please run: pip install pytorch3d")
            sys.exit(1)

        # knn_points 需要 batch 维度 (B, N, D)，这里 B=1
        knn_res = knn_points(
            xyz_b.unsqueeze(0),          # query (1, N, 3)
            xyz_b.unsqueeze(0),          # reference (1, N, 3)
            K=k_search,          # 包含自身
            return_sorted=True
        )
        knn_b = knn_res.idx.squeeze(0)[:, 1:k_search]   # (N, K+1)

        N, K = knn.shape

        # ------------------------------------------------------------
        # 1) boundary prefilter by cross-cluster ratio r_i + quality
        # ------------------------------------------------------------
        # is_cross: (N,K) where neighbor cluster differs
        # r_i: (N,)
        # boundary_mask: (N,)

        boundary_mask = self._boundary_by_cross_ratio(
            knn=knn_b, cluster_id=cluster_id_b, residual=residual_b
        )
        idx = torch.nonzero(boundary_mask).squeeze(1)     # (Nb,)
        if idx.numel() == 0:
            return idx.new_empty((0,)), idx.new_empty((0,)), xyz.new_empty((0,3)), self._opacity.new_empty((0,))

        # ------------------------------------------------------------
        # 2) build packed cross-cluster pairs (edges) for candidates
        #    no Kc_max: keep all cross neighbors
        # ------------------------------------------------------------
        # I,J: (E,) packed list of (i -> j) where i in idx and cid(j)!=cid(i)
        I, J = self._pack_cross_pairs(idx=idx, knn=knn_b, cluster_id=cluster_id_b)
        if I.numel() == 0:
            return idx.new_empty((0,)), idx.new_empty((0,)), xyz.new_empty((0,3)), self._opacity.new_empty((0,))
        # mids: (E,3)
        I = boundary_idx[I]
        J = boundary_idx[J]
        idx = boundary_idx[idx]
        mids = 0.5 * (xyz[I] + xyz[J])

        # ------------------------------------------------------------
        # 3) priority + base selection via K×Kc maximin over mids
        #    For each edge e=(I[e],J[e]) with mid=mids[e]:
        #      C_e = min_{p in P_{I[e]}} D(mids[e], p)
        #    Then per point i:
        #      priority[i] = max_{e: I[e]=i} C_e
        #      best_edge[i] = argmax_{e: I[e]=i} C_e
        # ------------------------------------------------------------

        priority_idx, best_edge_idx = self._compute_Ce2_chunked_and_segment_max(
            mids=mids, I=I, idx=idx, knn=knn, xyz=xyz, normal=normal
        )
        base_mid = mids[best_edge_idx]   # (Nb,3)
        pair_j   = J[best_edge_idx]      # (Nb,)

        # ------------------------------------------------------------
        # 4) choose which candidates to actually insert (budget a%)
        # ------------------------------------------------------------
        sel_local = self._select_by_priority(priority_idx, N)  # indices into [0..Nb-1], shape (M,)
        if sel_local.numel() == 0:
            return idx.new_empty((0,)), idx.new_empty((0,)), xyz.new_empty((0,3)), self._opacity.new_empty((0,))

        sel_local = self._dedup_selected_pairs(
            idx=idx,
            pair_j=pair_j,
            sel_local=sel_local,
            total_N=N,
        )

        sel_i = idx[sel_local]           # (M,)
        sel_j = pair_j[sel_local]        # (M,)
        b     = base_mid[sel_local]      # (M,3)     
        global_sel_i = mask_points[sel_i] 
        opacity_inter = self._opacity[global_sel_i]
        return sel_i, sel_j, b, opacity_inter

    @torch.no_grad()
    def _boundary_by_cross_ratio(
        self,
        knn: torch.Tensor,         # (N,K) long, all >= 0
        cluster_id: torch.Tensor,  # (N,)  long/int
        residual: torch.Tensor,    # (N,)  float
    ):
        """
        Boundary prefilter:
        - cross-cluster neighbor ratio r_i
        - residual threshold to remove abnormal / unreliable points
        - require at least one cross-cluster neighbor

        Returns:
            boundary_mask: (N,) bool
        """
        N, K = knn.shape

        # neighbor cluster ids: (N,K)
        cid_nb = cluster_id[knn]
        cid_center = cluster_id.view(N, 1)

        # cross-cluster indicator: (N,K)
        is_cross = (cid_nb != cid_center) & (cid_nb != -1) & (cid_center != -1)

        # r_i = (#cross)/K : (N,)
        r_i = is_cross.sum(dim=1, dtype=residual.dtype) / float(K)

        # thresholds
        ratio_thr = self.inter_ratio_thr
        # res_th = torch.quantile(residual, self.residual_th)

        # boundary_mask = (r_i >= ratio_thr) & (residual <= res_th)
        boundary_mask = (r_i >= ratio_thr)
        return boundary_mask

    @torch.no_grad()
    def _pack_cross_pairs(
        self,
        idx: torch.Tensor,         # (Nb,) long, boundary point indices
        knn: torch.Tensor,         # (N,K) long
        cluster_id: torch.Tensor,  # (N,) long/int
    ):
        """
        Build packed cross-cluster edges for boundary candidates.

        Returns:
            I, J: (E,), packed directed edges (i -> j), i in idx and cid(j) != cid(i)
        """

        # neighbors of boundary points: (Nb,K)
        nb = knn[idx]  # (Nb,K)

        # center cluster id: (Nb,1), neighbor cluster id: (Nb,K)
        cid_i = cluster_id[idx].view(-1, 1)
        cid_j = cluster_id[nb]

        # cross-cluster mask: (Nb,K)
        cross = (cid_j != cid_i) & (cid_j != -1)

        if not torch.any(cross):
            empty = knn.new_empty((0,), dtype=torch.long)
            return empty, empty

        # expand to edge list
        # I_all: (Nb,K) each row filled with idx[row]
        I_all = idx.view(-1, 1).expand(-1, 64)
        I = I_all[cross]   # (E,)
        J = nb[cross]      # (E,)

        return I, J


    @torch.no_grad()
    def _compute_Ce2_chunked_and_segment_max(
        self,
        mids: torch.Tensor,   # (E,3)
        I: torch.Tensor,      # (E,)
        idx: torch.Tensor,    # (S,)
        knn: torch.Tensor,    # (N,K)
        xyz: torch.Tensor,    # (N,3)
        normal: torch.Tensor, # (N,3)
    ):
        device = xyz.device
        dtype = xyz.dtype
        E = I.numel()
        S = idx.numel()

        if E == 0 or S == 0:
            priority_idx = torch.full((S,), 0.0, device=device, dtype=dtype)
            best_edge_idx = torch.full((S,), -1, device=device, dtype=I.dtype)
            return priority_idx, best_edge_idx

        # grouped source info
        src_u, counts = torch.unique_consecutive(I, return_counts=True)
        assert src_u.numel() == S and torch.equal(src_u, idx), \
            "[InterCluster] src_u must exactly match idx"

        rho = self.inter_rho
        K = knn.shape[1]
        K1 = K + 1

        # --------------------------------------------------
        # 1) source neighborhood
        # --------------------------------------------------
        nb_ext_all = torch.empty((S, K1), device=device, dtype=knn.dtype)
        nb_ext_all[:, 0] = idx
        nb_ext_all[:, 1:] = knn[idx]

        p_all = xyz[nb_ext_all]          # (S, K+1, 3)
        n_all = normal[nb_ext_all]       # (S, K+1, 3)
        n_src = normal[idx]              # (S, 3)

        dot_all = (n_src[:, None, :] * n_all).sum(dim=-1)  # (S, K+1)
        w_all = (2.0 - dot_all).clamp_min(0.0).pow(rho)    # (S, K+1)

        # edge -> source row id
        src_pos_for_edge = torch.repeat_interleave(
            torch.arange(S, device=device, dtype=I.dtype), counts
        )  # (E,)
        edge_ids = torch.arange(E, device=device, dtype=I.dtype)

        # --------------------------------------------------
        # 2) single pass: chunk-local max/argmax -> global merge
        # --------------------------------------------------
        priority_idx = torch.full((S,), -torch.inf, device=device, dtype=dtype)
        best_tmp = torch.full((S,), E, device=device, dtype=I.dtype)  # sentinel
        chunk_size = self.inter_edge_chunk_size

        for s in range(0, E, chunk_size):
            t = min(s + chunk_size, E)

            mids_blk = mids[s:t]                # (B,3)
            src_blk = src_pos_for_edge[s:t]     # (B,)
            edge_blk = edge_ids[s:t]            # (B,)

            p_blk = p_all[src_blk]              # (B,K+1,3)
            n_blk = n_all[src_blk]              # (B,K+1,3)
            w_blk = w_all[src_blk]              # (B,K+1)

            # ce2 for this chunk
            v = mids_blk[:, None, :] - p_blk
            proj = (v * n_blk).sum(dim=-1)
            v2 = (v * v).sum(dim=-1)
            d2 = (v2 - proj * proj).clamp_min_(0.0)
            ce2_blk = (d2 * w_blk).min(dim=1).values   # (B,)

            # ----------------------------------------------
            # chunk-local grouped max
            # ----------------------------------------------
            blk_max = torch.full((S,), -torch.inf, device=device, dtype=dtype)
            blk_max.scatter_reduce_(
                0, src_blk, ce2_blk, reduce="amax", include_self=True
            )

            # 找出 chunk 内达到局部最大值的 edge，并选最小 edge index
            hit = ce2_blk == blk_max[src_blk]
            cand = torch.where(hit, edge_blk, torch.full_like(edge_blk, E))

            blk_best = torch.full((S,), E, device=device, dtype=I.dtype)
            blk_best.scatter_reduce_(
                0, src_blk, cand, reduce="amin", include_self=True
            )

            # ----------------------------------------------
            # merge chunk result into global result
            # ----------------------------------------------
            better = blk_max > priority_idx
            equal = blk_max == priority_idx

            # 更大：直接替换 value 和 index
            priority_idx = torch.where(better, blk_max, priority_idx)
            best_tmp = torch.where(better, blk_best, best_tmp)

            # 相等：按 tie-break 选更小 edge index
            tie_update = equal & (blk_best < best_tmp)
            best_tmp = torch.where(tie_update, blk_best, best_tmp)

        # 与你原版保持一致：无边 source 的优先级置 0
        priority_idx.masked_fill_(priority_idx == -torch.inf, 0.0)
        best_edge_idx = torch.where(best_tmp == E, -1, best_tmp)

        return priority_idx, best_edge_idx


    def _select_by_priority(
        self,
        priority: torch.Tensor,   # (Nb,) float, higher is better
        N: int,             # N
    ):
        """
        Select top-M candidates by priority.

        Budget:
            Nb = priority.numel()
            M = clamp(round(alpha*Nb), M_min, round(beta*N))

        Returns:
            sel_local: (M,) long indices into [0..Nb-1]
        """
        Nb = int(priority.numel())
        if Nb == 0:
            return priority.new_empty((0,), dtype=torch.long)

        # hyperparams
        alpha = self.inter_topk_ratio

        # ---- compute budget ----
        M = int(round(alpha * Nb))

        # handle degenerate cases
        if M <= 0:
            return priority.new_empty((0,), dtype=torch.long)
        M = min(M, Nb)  # topk requires k <= Nb

        # topk returns indices into [0..Nb-1]
        _, sel_local = torch.topk(priority, k=M, largest=True, sorted=False)
        return sel_local

    @torch.no_grad()
    def _dedup_selected_pairs(
        self,
        idx: torch.Tensor,        # (Nb,)
        pair_j: torch.Tensor,     # (Nb,)
        sel_local: torch.Tensor,  # (M,)
        total_N: int,             # N, used to build unique key
    ):
        """
        Deduplicate selected inter-cluster pairs by undirected (i,j).

        Args:
            idx:       (Nb,) source indices
            pair_j:    (Nb,) paired neighbor indices
            sel_local: (M,) indices into [0..Nb-1]
            total_N:   total number of points N (for key construction)

        Returns:
            sel_local_dedup: (M',) subset of sel_local after deduplication
        """
        if sel_local.numel() == 0:
            return sel_local

        # gather selected pairs
        i = idx[sel_local]
        j = pair_j[sel_local]

        # undirected pair key
        key = i * int(total_N) + j

        # keep first occurrence of each key
        uniq, inv = torch.unique(key, return_inverse=True)  # uniq: (U,), inv: (M,) in [0..U-1]
        M = key.numel()
        U = uniq.numel()
        idx_in_M = torch.arange(M, device=key.device, dtype=torch.long)
        rep = torch.full((U,), M, device=key.device, dtype=torch.long)
        rep.scatter_reduce_(0, inv, idx_in_M, reduce="amin", include_self=True)  # (U,)
        keep = rep

        # return deduplicated sel_local
        return sel_local[keep]

    def _compute_inter_para(
        self,
        xyz: torch.Tensor,      # (M,3) midpoints
        sel_i: torch.Tensor,    # (M,) selected source indices
        sel_j: torch.Tensor,    # (M,) selected paired indices
        b: torch.Tensor,        # (M,3) midpoints
        normal: torch.Tensor,   # (N,3)
        knn: torch.Tensor,      # (N,K)
        dis2: torch.Tensor
    ):
        """
        Returns:
            xyz_inter:(M,3)    new center
            R:        (M,3,3) local frame
            scale:    (M,3)   Gaussian scales
        """
        if sel_i.numel() == 0:
            empty_xyz = self._xyz.new_empty((0, 3))
            empty_R = self._xyz.new_empty((0, 3, 3))
            empty_s = self._xyz.new_empty((0, 3))
            return empty_xyz, empty_R, empty_s


        # ------------------------------------------------
        # 1) for each pair, evaluate dk(b, n) with n_i and n_j
        #    choose direction with smaller |delta|
        # ------------------------------------------------
        delta_i = self._compute_weighted_offset(
            xyz,
            b=b,
            ref_normal=normal[sel_i],
            Nb_list=knn,
            normal=normal,
        )

        delta_j = self._compute_weighted_offset(
            xyz,
            b=b,
            ref_normal=normal[sel_j],
            Nb_list=knn,
            normal=normal,
        )

        # choose better direction
        choose_i = torch.abs(delta_i) <= torch.abs(delta_j)
        delta = torch.where(choose_i, delta_i, delta_j)
        n_ref = torch.where(
            choose_i[:, None],
            normal[sel_i],
            normal[sel_j]
        )
        xyz_inter = b + delta[:, None] * n_ref
        # ------------------------------------------------
        # 2) construct local frame & scales
        # ------------------------------------------------
        R_inter = self._build_local_frame(n_ref)
        scale = self._init_gaussian_scale(dis2)

        return xyz_inter, R_inter, scale


    @torch.no_grad()
    def _compute_weighted_offset(
        self,
        xyz: torch.Tensor,          # (M,3) midpoints
        b: torch.Tensor,            # (M,3) midpoints
        ref_normal: torch.Tensor,   # (M,3) corresponding normals n_i or n_j (unit)
        Nb_list: torch.Tensor,      # (M,K) neighborhood indices
        normal: torch.Tensor,       # (N,3)
    ):
        """
        Compute weighted offset dk(b, n) for each pair.

        Returns:
            delta: (M,) signed offsets along ref_normal
        """
        # hyper-parameters
        cos_th = math.cos(math.radians(self.max_angle_deg))
        eps = 1e-8

        # ------------------------------------------------
        # gather neighborhood points and normals
        # ------------------------------------------------
        # p_nb, n_nb: (M,K,3)
        p_nb = xyz[Nb_list]
        n_nb = normal[Nb_list]

        # ------------------------------------------------
        # geometry terms
        # ------------------------------------------------
        # v = b - p : (M,K,3)
        v = b[:, None, :] - p_nb

        # projection along ref_normal: (M,K)
        proj = (v * ref_normal[:, None, :]).sum(dim=-1)

        # distance ||b - p||: (M,K)
        r2 = (v * v).sum(dim=-1)
        sigma_p2 = torch.quantile(r2, q=0.8, dim=1, keepdim=True)
        theta = torch.exp(-r2 / (sigma_p2 + eps))

        # angular weight psi(n, n_p): (M,K)
        cos_nn = (ref_normal[:, None, :] * n_nb).sum(dim=-1)
        # numerical safety
        cos_nn = cos_nn.clamp(-1.0, 1.0)
        psi = torch.exp(-(((1.0 - cos_nn) / (1.0 - cos_th)) ** 2))

        # ------------------------------------------------
        # weighted average
        # ------------------------------------------------
        w = theta * psi                          # (M,K)
        num = (proj * w).sum(dim=1)              # (M,)
        den = w.sum(dim=1).clamp_min(eps)        # (M,)

        delta = num / den                        # (M,)
        return delta

    @torch.no_grad()
    def _build_local_frame(self, n: torch.Tensor) -> torch.Tensor:
        """
        Build an orthonormal frame from normal n.

        Args:
            n: (M,3) normals (not necessarily unit)

        Returns:
            R: (M,3,3) rotation matrices, columns are [t1, t2, n_unit]
        """
        n_unit = torch.nn.functional.normalize(n, dim=-1)

        # pick a reference axis not parallel to n
        ex = n_unit.new_tensor([1.0, 0.0, 0.0]).expand_as(n_unit)
        ey = n_unit.new_tensor([0.0, 1.0, 0.0]).expand_as(n_unit)

        use_ey = (n_unit.abs()[:, 0] > 0.9)  # if too aligned with x, use y
        ref = torch.where(use_ey[:, None], ey, ex)  # (M,3)

        t1 = torch.nn.functional.normalize(torch.cross(n_unit, ref, dim=-1), dim=-1)
        t2 = torch.nn.functional.normalize(torch.cross(n_unit, t1, dim=-1), dim=-1)

        R = torch.stack([n_unit, t1, t2], dim=-1)  # (M,3,3), columns
        return R

    @torch.no_grad()
    def _init_gaussian_scale(
        self,
        dis2: torch.Tensor,      # (N,K)
    ) -> torch.Tensor:
        """
        Initialize Gaussian scale in local frame:
            scale = [s_t, s_t, s_n], with s_n = ratio * s_t

        s_t is estimated from average neighbor distance.
        """
        c_n = self.inter_scale_n_ratio
        dist = torch.sqrt(dis2)  # (M,K)

        # robust-ish central tendency: mean (or you can swap to median/quantile later)
        s_t = (dist.mean(dim=1)) / 2                  # (M,)

        s_n = c_n * s_t                               # (M,)

        scale = torch.stack([s_n, s_t, s_t], dim=-1)  # (M,3)
        return scale


    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts

    def create_colored_pointcloud(
        self,
        points: torch.Tensor,
        cluster_id: torch.Tensor,
        cmap_name: str = "tab20",
        noise_color: list = [0.5, 0.5, 0.5],
        random_seed: int = 42,
        min_brightness: float = 0.3
    ):
        """
        根据 cluster_id 为点云生成 Open3D PointCloud 对象（带颜色）。
        
        参数:
            points: torch.Tensor，形状 (N, C)，C >= 3，前三维必须是 XYZ 坐标
            cluster_id: np.ndarray 或 torch.Tensor，形状 (N,)，簇标签（从 0 开始，-1 表示噪声）
            cmap_name: 颜色映射名称
                - "tab20": 推荐，离散颜色，最多 20 个簇时对比度最好
                - "hsv": 簇很多时使用，循环色相
                - "random": 随机生成颜色（带最小亮度避免太暗）
            noise_color: 噪声点 (-1) 的颜色，[R, G, B]，范围 0-1
            random_seed: 随机颜色时的种子
            min_brightness: 随机颜色时最小亮度
        
        返回:
            open3d.geometry.PointCloud 已着色好的点云对象，可直接可视化
        """
        # 转换为 numpy
        import matplotlib.pyplot as plt
        if torch.is_tensor(points):
            points_np = points.cpu().numpy()
        else:
            points_np = np.asarray(points)
        
        cluster_id = np.asarray(cluster_id.to("cpu"))
        if torch.is_tensor(cluster_id):
            cluster_id = cluster_id.cpu().numpy()
        
        n_points = points_np.shape[0]
        
        # 取前三维作为坐标（兼容 xyz + normal/rgb 等情况）
        xyz = points_np[:, :3]
        
        # 初始化颜色数组
        colors = np.zeros((n_points, 3))
        
        # 处理噪声点
        noise_mask = cluster_id == -1
        colors[noise_mask] = noise_color
        
        # 有效点掩码
        valid_mask = ~noise_mask
        if not valid_mask.any():
            # 全是噪声
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd
        
        valid_labels = cluster_id[valid_mask]
        num_clusters = int(valid_labels.max() + 1)
        
        if cmap_name == "random":
            # 随机颜色（可重复）
            np.random.seed(random_seed)
            cluster_colors = np.random.uniform(min_brightness, 1.0, size=(num_clusters, 3))
            colors[valid_mask] = cluster_colors[valid_labels]
        
        else:
            # 使用 matplotlib colormap
            cmap = plt.get_cmap(cmap_name)
            if num_clusters <= 20 and cmap_name == "tab20":
                norm_labels = valid_labels
            else:
                norm_labels = valid_labels / (num_clusters - 1 if num_clusters > 1 else 1)
            
            colors[valid_mask] = cmap(norm_labels)[:, :3]
        
        # 创建 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def visualize_pointcloud_with_color(self, xyz: torch.Tensor, color: torch.Tensor, title="pcd"):
        xyz_np = xyz.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy().clip(0.0, 1.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np)
        pcd.colors = o3d.utility.Vector3dVector(color_np)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 3.0
        vis.run()
        vis.destroy_window()

    def visualize_boundary_points(self, xyz: torch.Tensor, idx: torch.Tensor, title="boundary"):
        xyz_np = xyz.detach().cpu().numpy()
        N = xyz_np.shape[0]

        # color: default black
        color = np.zeros((N, 3), dtype=np.float32)

        # boundary points -> red
        idx_np = idx.detach().cpu().numpy()
        color[idx_np] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np)
        pcd.colors = o3d.utility.Vector3dVector(color)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 3.0
        vis.run()
        vis.destroy_window()

