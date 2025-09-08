# CS194-26 (CS294-26): Project 1 starter Python code
# You can use skimage / numpy; optionally matplotlib/opencv for I/O/vis.

import os
import numpy as np
import skimage as sk
import skimage.io as skio

from skimage.util import img_as_ubyte
from skimage.transform import rescale
from skimage.filters import sobel


# ---------------------------
# Basic utilities
# ---------------------------

def split_bgr_plate(im: np.ndarray):
    """将竖直拼接的底片（上到下：B, G, R）切成三个通道。"""
    H3 = (im.shape[0] // 3) * 3
    im = im[:H3, :]
    h = H3 // 3
    b = im[:h]
    g = im[h:2 * h]
    r = im[2 * h:3 * h]
    return b, g, r


def roll2(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """列方向平移 dx；行方向平移 dy（右、下为正）。"""
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def crop_interior(img: np.ndarray, border) -> np.ndarray:
    """
    支持像素或比例裁边：
    - int/np.integer => 按像素裁边
    - float/np.floating => 按短边比例裁边（例如 0.08 表示裁掉 8%）
    """
    if border is None:
        return img

    if isinstance(border, (int, np.integer)):
        b = int(border)
    else:  # float / numpy floating
        b = int(round(min(img.shape) * float(border)))

    if b <= 0:
        return img
    if img.shape[0] <= 2 * b or img.shape[1] <= 2 * b:
        # 避免裁没
        return img
    return img[b:-b, b:-b]


# ---------------------------
# Features & Metrics
# ---------------------------

def feature_grad(img: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值 + 归一化，抗亮度/对比度变化。"""
    g = sobel(img.astype(np.float32))
    m = g.max()
    return (g / m) if m > 1e-8 else g


def grad_mag(img: np.ndarray) -> np.ndarray:
    """简易梯度幅值特征（对亮度/对比度变化更稳健）。"""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:] = img[:, 1:] - img[:, :-1]
    gy[1:, :] = img[1:, :] - img[:-1, :]
    g = np.hypot(gx, gy)
    m = g.max()
    return g / m if m > 0 else g


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean NCC（越大越好）。"""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float((a * b).sum() / denom)


def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    """标准 NCC（越大越好）。"""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float((a * b).sum() / denom)


# ---------------------------
# Local search alignment
# ---------------------------

def align_local_search(
    ref: np.ndarray,
    mov: np.ndarray,
    center_dx: int = 0,
    center_dy: int = 0,
    radius: int = 8,
    border=0.08,
    use_edges: bool = True,
    metric: str = 'zncc',
):
    """在 (center_dx, center_dy) 附近半径=radius 的小窗口里局部搜索。"""
    R = feature_grad(ref) if use_edges else ref

    best_dx, best_dy = center_dx, center_dy
    best_score = -1e9  # 统一为“越大越好”

    for dy in range(center_dy - radius, center_dy + radius + 1):
        for dx in range(center_dx - radius, center_dx + radius + 1):
            M = roll2(mov, dx, dy)
            Mr = feature_grad(M) if use_edges else M

            Rc = crop_interior(R, border)
            Mc = crop_interior(Mr, border)
            if Rc.shape != Mc.shape or Rc.size == 0:
                continue

            if metric == 'zncc':
                s = zncc(Rc, Mc)
            else:
                # 默认为 zncc，如要用 L2，可改成负的 L2（越大越好）
                s = zncc(Rc, Mc)

            if s > best_score:
                best_score, best_dx, best_dy = s, dx, dy

    return best_dx, best_dy, best_score


def align(
    moving: np.ndarray,
    reference: np.ndarray,
    search_radius: int = 15,
    border=20,
    use_edges: bool = False,
    metric: str = "ncc",
):
    """
    单尺度穷举平移对齐：
    - 在 [-search_radius, search_radius] 内移动
    - 用 NCC / L2 评估，返回最佳对齐后的图与偏移
    """
    Ref = grad_mag(reference) if use_edges else reference
    Ref = crop_interior(Ref, border)

    best_score = -1e9 if metric == "ncc" else 1e9
    best_dx, best_dy = 0, 0

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            M = roll2(moving, dx, dy)
            Mov = grad_mag(M) if use_edges else M
            Mov = crop_interior(Mov, border)

            if Mov.shape != Ref.shape or Ref.size == 0:
                continue

            if metric == "ncc":
                s = ncc_score(Ref, Mov)
                better = s > best_score
            else:
                diff = Ref - Mov
                s = float((diff * diff).sum())
                better = s < best_score

            if better:
                best_score, best_dx, best_dy = s, dx, dy

    aligned = roll2(moving, best_dx, best_dy)
    print(f"[single] Best offset: dx={best_dx}, dy={best_dy}")
    return aligned, (best_dx, best_dy)


# ---------------------------
# Pyramid alignment
# ---------------------------

def build_pyramid(
    img: np.ndarray,
    scale: float = 0.5,
    min_side: int = 128,
    max_levels: int | None = None,
):
    """返回从粗到细（coarse->fine）的金字塔列表。"""
    pyr = [img.astype(np.float32)]
    while True:
        if max_levels is not None and len(pyr) >= max_levels:
            break
        h, w = pyr[-1].shape
        if min(h, w) <= min_side:
            break
        down = rescale(
            pyr[-1],
            scale,
            anti_aliasing=True,
            channel_axis=None,
            preserve_range=True,
        ).astype(np.float32)
        pyr.append(down)
    return pyr[::-1]  # 粗 -> 细


def align_pyramid(
    ref: np.ndarray,
    mov: np.ndarray,
    scale: float = 0.5,
    min_side: int = 128,
    max_levels: int | None = None,
    coarse_radius: int = 40,
    fine_radius: int = 8,
    border_ratio: float = 0.08,
    use_edges: bool = True,
    metric: str = 'zncc',
):
    """
    更稳的金字塔对齐：
      1) 建金字塔从粗到细
      2) 最粗层：以 (0,0) 为中心，半径=coarse_radius 搜索
      3) 到更细一层：把上层位移 * (1/scale)（scale=0.5 → ×2），作为中心再细化
      4) border 按比例裁边（默认 8%），随分辨率自动变化

    返回：对齐后的 mov 以及相对 ref 的 (dx, dy)（在原图坐标系）
    """
    # 1) 建金字塔（粗->细）
    R_pyr = build_pyramid(ref, scale=scale, min_side=min_side, max_levels=max_levels)
    M_pyr = build_pyramid(mov, scale=scale, min_side=min_side, max_levels=max_levels)
    assert len(R_pyr) == len(M_pyr)
    L = len(R_pyr)

    # 2) 在最粗层搜索
    dx, dy = 0, 0
    R_coarse, M_coarse = R_pyr[0], M_pyr[0]
    dx, dy, _ = align_local_search(
        R_coarse, M_coarse,
        center_dx=0, center_dy=0,
        radius=coarse_radius,
        border=border_ratio,
        use_edges=use_edges,
        metric=metric,
    )

    # 3) 逐层细化
    for lvl in range(1, L):
        # 将位移放大到当前层坐标（scale=0.5 → 每下到一层 ×2）
        scale_up = 1 / scale
        dx = int(round(dx * scale_up))
        dy = int(round(dy * scale_up))

        Rl, Ml = R_pyr[lvl], M_pyr[lvl]
        dx, dy, _ = align_local_search(
            Rl, Ml,
            center_dx=dx, center_dy=dy,
            radius=fine_radius,
            border=border_ratio,
            use_edges=use_edges,
            metric=metric,
        )

    aligned = roll2(mov, dx, dy)
    return aligned, (dx, dy)


# ---------------------------
# Pipeline
# ---------------------------

def colorize(imname: str):
    data_root = './data'
    impath = os.path.join(data_root, imname)

    im = skio.imread(impath)          # jpg: uint8; tif: uint16/float
    im = sk.img_as_float(im)          # -> float [0,1]

    # 切分 B/G/R
    b, g, r = split_bgr_plate(im)

    ext = os.path.splitext(imname)[1].lower()
    is_tif = ext in ('.tif', '.tiff')

    if is_tif:
        # 大图/难例（如 emir.tif）：金字塔 + 梯度幅值
        ag, g_off = align_pyramid(
            b, g,
            scale=0.5,
            min_side=128,
            max_levels=None,
            coarse_radius=40,
            fine_radius=8,
            border_ratio=0.08,
            use_edges=True,
            metric='zncc',
        )
        ar, r_off = align_pyramid(
            b, r,
            scale=0.5,
            min_side=128,
            max_levels=None,
            coarse_radius=40,
            fine_radius=8,
            border_ratio=0.08,
            use_edges=True,
            metric='zncc',
        )
    else:
        # 小图 / 低分辨率 jpg：单尺度
        ag, g_off = align(g, b, search_radius=15, border=20, use_edges=False, metric="ncc")
        ar, r_off = align(r, b, search_radius=15, border=20, use_edges=False, metric="ncc")

    # 组合为 RGB（R, G, B）
    im_out = np.dstack([ar, ag, b])

    # 保存
    os.makedirs('./out', exist_ok=True)
    stem, _ = os.path.splitext(imname)
    outpath = os.path.join('./out', f'{stem}_color.jpg')
    skio.imsave(outpath, img_as_ubyte(np.clip(im_out, 0, 1)))

    print(f"Saved: {outpath}")
    print(f"Offsets (G,R relative to B): G{g_off}, R{r_off}")


# ---------------------------
# Entry
# ---------------------------

if __name__ == "__main__":
    data_dir = './data'
    for imname in os.listdir(data_dir):
        if imname.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff')):
            print(f'Processing {imname} ...')
            colorize(imname)
