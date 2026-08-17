import math
import numpy as np
import torch
import torch.nn.functional as F
from gaussian_renderer import render_normal
import matplotlib.pyplot as plt
import sys
# 从 pseudo_cam 恢复相机参数

def build_extrinsics_intrinsics_from_pseudo_cam(pseudo_cam):
    """
    根据 pseudo_cam.R / T / FoVx / FoVy / image_width / image_height
    恢复：
        extrinsics: (1, 4, 4)  world-to-camera
        intrinsics: (1, 3, 3)

    约定：
    - pseudo_cam.R: (3, 3)
    - pseudo_cam.T: (3,)
    - pseudo_cam.FoVx, FoVy: 弧度
    - pseudo_cam.image_width, pseudo_cam.image_height: 像素

    返回 np.float32
    """
    W = int(pseudo_cam.image_width)
    H = int(pseudo_cam.image_height)

    # ---- intrinsics ----
    # fx = W / (2 * tan(FoVx / 2))
    # fy = H / (2 * tan(FoVy / 2))
    fx = W / (2.0 * math.tan(float(pseudo_cam.FoVx) / 2.0))
    fy = H / (2.0 * math.tan(float(pseudo_cam.FoVy) / 2.0))
    cx = W / 2.0
    cy = H / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # ---- extrinsics ----
    # 通常 3DGS / COLMAP 风格是：
    #   X_cam = R @ X_world + T
    # 这里直接组 world-to-camera 4x4
    R = np.asarray(pseudo_cam.R, dtype=np.float32)
    T = np.asarray(pseudo_cam.T, dtype=np.float32).reshape(3)

    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = R
    ext[:3, 3] = T

    # 单张图，补 batch-like N 维
    extrinsics = torch.from_numpy(ext).to("cuda")  # (4, 4)
    intrinsics = torch.from_numpy(K).to("cuda")    # (3, 3)

    return extrinsics, intrinsics


def resize_intrinsics(K, H0, W0, H1, W1):
    sx = W1 / W0
    sy = H1 / H0
    K1 = K.clone()
    K1[..., 0, 0] *= sx
    K1[..., 1, 1] *= sy
    K1[..., 0, 2] *= sx
    K1[..., 1, 2] *= sy
    return K1

def estimate_depth_da3_diff(
    img_chw:torch.tensor,         # (3,H,W), torch.Tensor
    extrinsics:torch.tensor,      # (4,4), torch.Tensor
    intrinsics:torch.tensor,      # (3,3), torch.Tensor
    da3_model
):
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3

    H0, W0 = img_chw.shape[-2:]
    H1 = (H0 // 14) * 14
    W1 = (W0 // 14) * 14

    img_resized = F.interpolate(
        img_chw.unsqueeze(0),   # (1,3,H,W)
        size=(H1, W1),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)                # (3,H1,W1)

    K_resized = resize_intrinsics(intrinsics, H0, W0, H1, W1)

    image_batched = img_resized.unsqueeze(0).unsqueeze(0)     # (1,1,3,H1,W1)
    extr_batched  = extrinsics.unsqueeze(0).unsqueeze(0)      # (1,1,4,4)
    intr_batched  = K_resized.unsqueeze(0).unsqueeze(0)       # (1,1,3,3)

    out = da3_model.model(
        image_batched,
        extr_batched,
        intr_batched,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="first",
    )

    depth = F.interpolate(
        out["depth"],
        size=(H0, W0),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # if hasattr(depth, "detach"):   # torch.Tensor
    #     depth_vis = depth.detach().cpu().numpy()
    # else:
    #     depth_vis = depth

    # plt.figure()
    # plt.imshow(depth_vis, cmap="viridis")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig("depth_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0)
    # plt.close()
    # sys.exit(0)

    return depth



# DA3 深度估计函数
def estimate_depth_da3(img: torch.tensor, pseudo_cam, mode="test", model=None):
    """
    使用 DA3-LARGE-1.1 估计单张图深度
    """

    extrinsics, intrinsics = build_extrinsics_intrinsics_from_pseudo_cam(pseudo_cam)

    if mode == "test":
        with torch.no_grad():
            depth = estimate_depth_da3_diff(img, extrinsics, intrinsics, model)
    else:
        depth = estimate_depth_da3_diff(img, extrinsics, intrinsics, model)

    depth_normal = render_normal(pseudo_cam, depth)

    return depth, depth_normal
