import glob
import os
import math
from PIL import Image
import numpy as np
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data


class BatchManager:
    def __init__(self, data, batch_size):
        # 打乱data
        np.random.shuffle(data)
        self.data = data
        self.batch_size = batch_size
        self.batch_imgs = None
        self.idx = 0
        self.max_idx = len(data) // batch_size

    def next_batch(self):
        if not self.idx < self.max_idx:
            self.idx = 0
            # 打乱data
            np.random.shuffle(self.data)

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
