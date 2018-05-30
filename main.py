from utils import *
import tensorflow as tf
from model import DCGAN
from dcgan_old import DCGAN as old_dcgan

if __name__ == '__main__':
    imgs = load_anime_faces()
    # 开始训练
    with tf.Session() as sess:
        dcgan = DCGAN(sess, height=64, width=64, channel=3, checkpoint_dir='./checkpoint_anime8',
                      sample_dir='./sample_dir_anime8_30', sample_size=48)
        dcgan.train(imgs)
        # dcgan.sample('smple_test',70)
