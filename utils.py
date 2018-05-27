import glob
import os
import math
from PIL import Image
import numpy as np
from scipy import misc


def save_images(images, sample_dir, epoch=0, idx=0, sample_size=64):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # 用生成的图片数据生成 PNG 图片
    images = np.squeeze(images)
    for i in range(sample_size):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("{}/image-epoch{}-idx{}-{}.png".format(sample_dir, epoch, idx, i))


def load_anime_faces():
    # 获取训练数据
    images = []
    for image in glob.glob("faces/*"):
        image_data = misc.imread(image)  # imread 利用 PIL 来读取图片数据
        images.append(image_data)
    images = np.array(images)

    # 将数据标准化成 [-1, 1] 的取值
    input_data = (images.astype(np.float32) - 127.5) / 127.5
    return input_data


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def get_files(path):
    return list(glob.glob('{}/*'.format(path)))


def get_image_from_file(image_path):
    img = misc.imread(image_path)
    # 返回[-1,1]之间的图片数据
    return np.array(img) / 127.5 - 1.


if __name__ == '__main__':
    print(get_image_from_file(get_files('faces')[0]))
