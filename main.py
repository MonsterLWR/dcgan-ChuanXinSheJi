from utils import *
import tensorflow as tf
from dcgan import DCGAN

if __name__ == '__main__':
    # imgs = load_anime_faces()
    # 开始训练
    with tf.Session() as sess:
        dcgan = DCGAN(sess, height=64, width=64, channel=3, checkpoint_dir='./check_points/checkpoint_face01',
                      sample_dir='./sample/face01_04', sample_size=36)
        # for i in range(20):
        #     dir = './final_cuted_faces/faces_{}'.format(i)
        #     dcgan.train(dir)
        dir = './final_cuted_faces'
        dcgan.train(dir)
    # dcgan.complete(imgs[0:64, :, :, :], mask_type='center', out_dir='complete_test_2')
    # dcgan.sample('sample_test_minist_5', 60)
