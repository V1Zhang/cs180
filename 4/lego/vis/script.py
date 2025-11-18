#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image

# 根目录：lego/vis
ROOT = "/Users/v1zhang/交流/ucb bga/cs180/4/lego/vis"
RENDER_DIR = os.path.join(ROOT, "render")

# 支持的图片格式
EXTS = {".png"}

def is_image(fname):
    return os.path.splitext(fname.lower())[1] in EXTS


def resize_and_crop(img, target_w, target_h):
    """
    1. 先统一高度到 target_h（比例缩放）
    2. 再按 target_w 居中裁剪宽度
    """

    w, h = img.size

    # --- Step 1: 调整高度到 target_h（按比例缩放宽度）---
    scale = target_h / h
    new_w = int(w * scale)
    img = img.resize((new_w, target_h), Image.LANCZOS)

    # --- Step 2: 居中裁切宽度 ---
    if new_w >= target_w:
        left = (new_w - target_w) // 2
        right = left + target_w
        img = img.crop((left, 0, right, target_h))
    else:
        # 宽度不够 -> 居中填充（补黑边）
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        x = (target_w - new_w) // 2
        new_img.paste(img, (x, 0))
        img = new_img

    return img


def main():
    files = sorted(f for f in os.listdir(RENDER_DIR) if is_image(f))
    if not files:
        print("❌ render 目录下没有找到任何图片！")
        return

    base_path = os.path.join(ROOT, "ray_samples.png")
    if not os.path.exists(base_path):
        print(f"❌ 找不到基准图片: {base_path}")
        return

    base_img = Image.open(base_path)
    target_w, target_h = base_img.size
    print("Target size:", (target_w, target_h))

    for fname in files:
        img_path = os.path.join(RENDER_DIR, fname)
        try:
            img = Image.open(img_path)
            print("Before:", fname, img.size)

            fixed = resize_and_crop(img, target_w, target_h)
            fixed.save(img_path)

            print(f"✔ Processed {fname} → {fixed.size}")

        except Exception as e:
            print(f"⚠ 跳过 {fname}: {e}")

    print("\n🎉 全部裁剪完毕！")


if __name__ == "__main__":
    main()
