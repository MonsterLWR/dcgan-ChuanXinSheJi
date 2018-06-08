import glob
import os
import math

import scipy
from PIL import Image
import numpy as np
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data


class BatchManager:
    def __init__(self, files, batch_size):
        # files 训练图片数据的文件名
        # dir, _ = os.path.split(files[0])
        self.data = []
        for file in files:
            img = misc.imread(file)
            self.data.append(img)
        self.data = (np.array(self.data).astype(np.float32) - 127.5) / 127.5
        # 打乱data
        np.random.shuffle(self.data)
        self.batch_size = batch_size
        self.batch_imgs = None
        self.idx = 0
        self.max_idx = len(self.data) // batch_size
        # print('start training img in {}'.format(dir))

    def next_batch(self):
        if not self.idx < self.max_idx:
            self.idx = 0
            # 打乱data
            np.random.shuffle(self.data)

        # # 对应当前使用的100个batch的索引
        # temp_idx = self.idx % 100
        # if temp_idx == 0:
        #     # 存储当前使用的100个batch的数据
        #     self.data = []
        #
        #     # 读取100个batch的数据
        #     filenames = self.files[self.idx * self.batch_size: min((self.idx + 100), self.max_idx) * self.batch_size]
        #     self.data = get_image_from_files(self.files, filenames)
        #     print('new batch in {} from {} to {}.'.format(self.dir, self.idx, min((self.idx + 100), self.max_idx)))
        #     # self.data = (np.array(self.data).astype(np.float32) - 127.5) / 127.5

        self.batch_imgs = self.data[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]

        self.idx += 1

        return self.batch_imgs

    def cur_batch(self):
        return self.batch_imgs


def save_images(images, sample_dir, epoch=0, counter=0, sample_num=64):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # 用生成的图片数据生成 PNG 图片
    images = np.squeeze(images)
    for i in range(sample_num):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save(
            "{}/image-epoch{}-counter{}-{}.png".format(sample_dir, epoch, counter, i))


def merge_and_save(images, size, path, filename, idx):
    img = merge_imgs(images, size)
    if not os.path.exists(path):
        os.makedirs(path)

    img = img * 127.5 + 127.5
    Image.fromarray(img.astype(np.uint8)).save("{}/image-idx{}-{}.png".format(path, idx, filename))


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def merge_imgs(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def load_minist():
    # 读取minist数据集
    minist = input_data.read_data_sets('./MINIST_data/')
    images = minist.test.images.reshape(10000, 28, 28, 1)
    # 将数据的范围从[0,1]转变成[-1,1]
    images = (images * 2) - 1
    return images


def load_anime_faces():
    # 获取训练数据
    images = []
    for image in glob.glob("anime_faces/*"):
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


def get_image_from_files(dirs, filenames):
    # 从dirs中通过一次遍历读取filenames包含的图像
    filenames = set(filenames)
    imgs = []
    for filename in dirs:
        if filename in filenames:
            img = misc.imread(filename)
            imgs.append(img)
    # 返回[-1,1]之间的图片数据
    # return np.array(imgs).astype(np.float32) / 127.5 - 1.
    return (np.array(imgs).astype(np.float32) - 127.5) / 127.5


if __name__ == '__main__':
    # print(get_image_from_file(get_files('faces')[0]))
    pass
