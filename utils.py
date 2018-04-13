import os
import math
from PIL import Image
import numpy as np


def save_images(images, sample_dir, epoch, idx, sample_size):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # 用生成的图片数据生成 PNG 图片
    images = np.squeeze(images)
    for i in range(sample_size):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("{}/image-epoch{}-idx{}-{}.png".format(sample_dir, epoch, idx, i))


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
