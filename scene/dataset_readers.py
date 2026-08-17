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
DIETNERF_8_TRAIN_IDS = [2, 16, 26, 55, 73, 75, 86, 93]
import cv2
from tqdm import tqdm
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# import open3d as o3d

# def fetchPly(path):

#     pcd_o3d = o3d.io.read_point_cloud(path)
    
#     if len(pcd_o3d.points) == 0:
#         raise ValueError(f"Empty point cloud in {path}")
    
#     pcd_o3d.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=16)
#     )
    
#     # # 统一法向量方向（指向外侧，Open3D 内置功能）
#     pcd_o3d.orient_normals_consistent_tangent_plane(k=30)
    
#     positions = np.asarray(pcd_o3d.points)                    # (N, 3)
#     colors    = np.asarray(pcd_o3d.colors)
#     normals   = np.asarray(pcd_o3d.normals)                   # (N, 3) 已归一化
    
#     print(f"[fetchPly] Loaded {len(positions)} points with estimated normals")
    
#     return BasicPointCloud(
#         points=positions.astype(np.float32),
#         colors=colors.astype(np.float32),
#         normals=normals.astype(np.float32)
#     )

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, n_views, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse_views/fused.ply")
    pcd = fetchPly(ply_path)
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path) or True:
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #         print(f"xyz {xyz.shape}")
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
#     cam_infos = []

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]

#         frames = contents["frames"]
#         for idx, frame in enumerate(frames):
#             cam_name = os.path.join(path, frame["file_path"] + extension)

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_path = os.path.join(path, cam_name)
#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)

#             im_data = np.array(image.convert("RGBA"))

#             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy 
#             FovX = fovx

#             cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                             image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
#     return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
            
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            width = image.size[0]
            height = image.size[1]
            fx = fov2focal(FovX, width)
            fy = fov2focal(FovY, height)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    global_id=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy
                )
            )

            # arr = cv2.resize(arr, (400, 400))
            # image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
            # focal = fov2focal(fovx, image.size[0])

            # cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image_path=image_path,
            #                             image_name=image_name, width=image.size[0], height=image.size[1], fx=focal, fy=focal))
    return cam_infos



def readNerfSyntheticInfo(path, white_background, eval, n_views = 0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views
    ply_path = os.path.join(path, "sparse_views/fused.ply")

    # if n_views > 0:
    #     if n_views == 8:
    #         train_cam_infos = [train_cam_infos[i] for i in DIETNERF_8_TRAIN_IDS]
    #     else:
    #         train_cam_infos = train_cam_infos[:n_views]
    # ply_path = os.path.join(path, "1/fused.ply")


    nerf_normalization = getNerfppNorm(train_cam_infos)

    need_random = False
    print(ply_path)
    if not os.path.exists(ply_path):
        need_random = True
        print(f"[INFO] fused.ply 不存在 → 将生成随机点云 (100_000 个点)")
    else:
        try:
            temp_pcd = fetchPly(ply_path)
            num_pts = temp_pcd.points.shape[0]
            if num_pts < 100:
                need_random = True
                print(f"[INFO] fused.ply 仅包含 {num_pts} 个点（< 1000）→ 将生成随机点云")
            else:
                print(f"[INFO] fused.ply 已存在且包含 {num_pts} 个点 → 直接使用")
        except Exception as e:
            need_random = True
            print(f"[WARN] 读取 fused.ply 失败 ({e}) → 回退生成随机点云")

    if need_random:
        # Since this data set has no colmap data / 点数太少, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        ply_path = os.path.join(path, "sparse_views/fused_random.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        raise ValueError(f"The pcd = fetchPly(ply_path) wrong: {e}")

    # ply_path = os.path.join(path, "sparse_views/fused.ply")
    # nerf_normalization = getNerfppNorm(train_cam_infos)

    # need_random = False
    # existing_pcd = None
    # existing_xyz = None
    # existing_rgb = None


    # if not os.path.exists(ply_path):
    #     need_random = True
    #     print(f"[INFO] fused.ply 不存在 → 将生成随机点云")
    # else:
    #     try:
    #         existing_pcd = fetchPly(ply_path)
    #         existing_xyz = np.asarray(existing_pcd.points)
    #         existing_rgb = np.asarray(existing_pcd.colors)

    #         num_existing = existing_xyz.shape[0]

    #         if num_existing < 1_000_000:
    #             need_random = True
    #             print(f"[INFO] fused.ply 包含 {num_existing} 个点（< 1,000,000）→ 将保留已有点并追加随机点")
    #         else:
    #             print(f"[INFO] fused.ply 已存在且包含 {num_existing} 个点 → 直接使用")

    #     except Exception as e:
    #         need_random = True
    #         existing_pcd = None
    #         existing_xyz = None
    #         existing_rgb = None
    #         print(f"[WARN] 读取 fused.ply 失败 ({e}) → 回退生成纯随机点云")

    # if need_random:
    #     # 固定额外生成 100,000 个随机点
    #     num_random = 100_000
    #     print(f"Generating random point cloud ({num_random})...")

    #     # Blender synthetic 默认随机范围：[-1.3, 1.3]^3
    #     random_xyz = np.random.random((num_random, 3)) * 2.6 - 1.3

    #     random_shs = np.random.random((num_random, 3)) / 255.0
    #     random_rgb = SH2RGB(random_shs)

    #     if existing_xyz is not None and existing_xyz.shape[0] > 0:
    #         print(f"[INFO] 拼接已有点云 {existing_xyz.shape[0]} + 随机点 {num_random}")

    #         # fetchPly 通常返回 [0, 1] RGB；如果不是，则做兼容处理
    #         if existing_rgb.max() > 1.5:
    #             existing_rgb_01 = existing_rgb / 255.0
    #         else:
    #             existing_rgb_01 = existing_rgb

    #         xyz = np.concatenate([existing_xyz, random_xyz], axis=0)
    #         rgb = np.concatenate([existing_rgb_01, random_rgb], axis=0)

    #     else:
    #         print(f"[INFO] 没有可用已有点云 → 使用纯随机点云 {num_random}")
    #         xyz = random_xyz
    #         rgb = random_rgb

    #     pcd = BasicPointCloud(
    #         points=xyz,
    #         colors=rgb,
    #         normals=np.zeros((xyz.shape[0], 3))
    #     )

    #     ply_path = os.path.join(path, "sparse_views/fused_random.ply")
    #     storePly(ply_path, xyz, rgb * 255.0)
    # else:
    #     pcd = existing_pcd


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}