from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from model import DCGAN

# 读取minist数据集
minist = input_data.read_data_sets('./MINIST_data/')
images = minist.test.images.reshape(10000, 28, 28, 1)
# 打乱数据集顺序
np.random.shuffle(images)
# 将数据的范围从[0,1]转变成[-1,1]
images = (images * 2) - 1

# 开始训练
with tf.Session() as sess:
    dcgan = DCGAN(sess, height=28, width=28, channel=1, checkpoint_dir='./checkpoint', sample_dir='./sample_dir')
    dcgan.train(images)
