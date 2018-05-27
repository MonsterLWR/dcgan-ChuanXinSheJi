from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from utils import *
import tensorflow as tf
from model import DCGAN

if __name__ == '__main__':
    # # 读取minist数据集
    # minist = input_data.read_data_sets('./MINIST_data/')
    # images = minist.test.images.reshape(10000, 28, 28, 1)
    # # 打乱数据集顺序
    # np.random.shuffle(images)
    # # 将数据的范围从[0,1]转变成[-1,1]
    # images = (images * 2) - 1

    # np.random.shuffle(images)
    # print(images[0].shape)

    # save_images(images, './shuffle_test_2')

    # imgs = load_anime_faces()
    # 开始训练
    with tf.Session() as sess:
        dcgan = DCGAN(sess, height=64, width=64, channel=3, checkpoint_dir='./checkpoint_anime3',
                      sample_dir='./sample_dir_anime3_30', sample_size=10)
        # dcgan.train(imgs)
        dcgan.sample('smple_test')
