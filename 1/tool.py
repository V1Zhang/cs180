from PIL import Image
import numpy as np

# 打开 tif
im = Image.open("data/emir.tif")

# 转 numpy 数组
arr = np.array(im)

# 把 16-bit 数据归一化到 0-255
arr8 = (arr / arr.max() * 255).astype(np.uint8)

# 转回 PIL 图像
im8 = Image.fromarray(arr8)

# 保存成 jpg
im8.save("media/emir.jpg", quality=92)
