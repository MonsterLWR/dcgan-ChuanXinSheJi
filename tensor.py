from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import DCGAN
import os
from PIL import Image


# def save_images(images, sample_dir, epoch, idx):
#     if not os.path.exists(sample_dir):
#         os.makedirs(sample_dir)
#
#     # 用生成的图片数据生成 PNG 图片
#     images = np.squeeze(images)
#     for i in range(10):
#         image = images[i] * 127.5 + 127.5
#         image = np.uint8(image)
#         print(image)
#         # print(image)
#         image = Image.fromarray(image)
#         image.save("{}/image-epoch{}-idx{}-{}.png".format(sample_dir, epoch, idx, i))


minist = input_data.read_data_sets('./MNIST_data/')
# print(minist.test.images.shape)
images = minist.test.images.reshape(10000, 28, 28, 1)
np.random.shuffle(images)
images = (images * 2) - 1
# print(images[0])
# print(images.shape)

# save_images(images, "./sample_dir", 0, 0)

# x = tf.constant([1.0, 1.0])
# y = tf.constant([1.0, 1.0])
# z = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
#
# sess = tf.Session()
# res = sess.run(z)
# print(res)

# with tf.variable_scope("foo") as scope:
#     v = tf.get_variable("v", [1])
#     # scope.reuse_variables()
#     v1 = tf.get_variable("v", [1])
# assert v1 == v

with tf.Session() as sess:
    dcgan = DCGAN(sess, height=28, width=28, channel=1, checkpoint_dir='./checkpoint', sample_dir='./sample_dir_2')
    dcgan.train(images)
