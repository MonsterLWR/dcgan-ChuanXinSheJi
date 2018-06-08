import tensorflow as tf
import time
from utils import *
import oprations as ops

# 超参数设定
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETA_1 = 0.5  # Momentum term of adam [0.5]

# 噪声向量的大小
Z_DIM = 100
# Dimension of discrim filters in first conv layer.
D_F_DIM = 64
# Dimension of gen filters in first conv layer.
G_F_DIM = 64

# WGAN
CLIP = [-0.01, 0.01]
CRITIC_NUM = 5


class DCGAN():
    def __init__(self, sess, height=64, width=64, channel=3, checkpoint_dir=None, sample_dir=None, sample_size=64):
        # tensorflow session
        self.sess = sess
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # 生成图片数量，默认为64张
        self.sample_size = sample_size

        # 获取图片的各个维度
        # channel 1代表灰度图 3代表RGB
        self.image_dims = {'height': height,
                           'width': width,
                           'channel': channel}

        # checkpoint_dir保存模型参数的文件
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # tensorflow placeholder, 用于填充训练数据
        self.inputs = tf.placeholder(
            tf.float32, [BATCH_SIZE] + list(self.image_dims.values()), name='real_images')

        # 产生图片的随机向量，None代表随机向量的个数不一定等于batch_size
        # 因为随机向量的个数表示生成图片的个数
        self.z = tf.placeholder(
            tf.float32, [None, Z_DIM], name='z')

        # 生成器和判别其的构建
        # G代表生成的图片,取值为[-1,1]
        self.G = self.generator(self.z)
        # D 代表判别器输入真实图片产生的输出
        _, self.D_logits = self.discriminator(self.inputs, reuse=False)
        # D_ 代表判别器输入生成图片产生的输出
        _, self.D_logits_ = self.discriminator(self.G, reuse=True)
        # 用于产生采样图片，与generator共享参数
        self.sample_img = self.sampler(self.z)

        # # 用于查看图片的判别其的输出
        # self.temp_img = tf.placeholder(
        #     tf.float32, [None,
        #                  self.image_dims['height'],
        #                  self.image_dims['width'],
        #                  self.image_dims['channel']
        #                  ], name='temp_img')
        # self.temp_D = self.discriminator_for_sample(self.temp_img)

        # 判别其对真实图片的损失函数
        self.d_loss_real = tf.reduce_mean(tf.scalar_mul(-1, self.D_logits))
        # tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # 判别其对生成图片的损失函数
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        # tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        # 生成器的损失函数
        self.g_loss = tf.reduce_mean(tf.scalar_mul(-1, self.D_logits_))
        # tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # 判别器的损失函数
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # Returns all variables created with trainable=True
        t_vars = tf.trainable_variables()

        # 分别得到生成器和判别其参数，用于使用optimizer时限制可更新的参数
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # 用于保存参数到硬盘的对象
        self.saver = tf.train.Saver()

    def generator(self, z):

        with tf.variable_scope("generator"):
            s_h, s_w = self.image_dims['height'], self.image_dims['width']
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # 将随机向量z作为输入，通过全连接生成[s_h16, s_w16, G_F_DIM * 8]大小的输出
            g_h0_lin = ops.linear(z, G_F_DIM * 8 * s_h16 * s_w16, 'g_h0_lin')
            g_h0_re = tf.reshape(g_h0_lin, [-1, s_h16, s_w16, G_F_DIM * 8])
            g_bn0 = ops.batch_norm(g_h0_re, name='g_bn0', train=self.is_train)
            h0 = tf.nn.relu(g_bn0)

            g_h1 = ops.deconv2d(h0, [BATCH_SIZE, s_h8, s_w8, G_F_DIM * 4], name='g_h1')
            g_bn1 = ops.batch_norm(g_h1, name='g_bn1', train=self.is_train)
            h1 = tf.nn.relu(g_bn1)

            g_h2 = ops.deconv2d(h1, [BATCH_SIZE, s_h4, s_w4, G_F_DIM * 2], name='g_h2')
            g_bn2 = ops.batch_norm(g_h2, name='g_bn2', train=self.is_train)
            h2 = tf.nn.relu(g_bn2)

            g_h3 = ops.deconv2d(h2, [BATCH_SIZE, s_h2, s_w2, G_F_DIM * 1], name='g_h3')
            g_bn3 = ops.batch_norm(g_h3, name='g_bn3', train=self.is_train)
            h3 = tf.nn.relu(g_bn3)

            h4 = ops.deconv2d(h3, [BATCH_SIZE, s_h, s_w, self.image_dims['channel']], name='g_h4')

            # 激活函数使用tanh
            return tf.nn.tanh(h4)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.image_dims['height'], self.image_dims['width']
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # 将随机向量z作为输入，通过全连接生成[s_h16, s_w16, G_F_DIM * 8]大小的输出
            g_h0_lin = ops.linear(z, G_F_DIM * 8 * s_h16 * s_w16, 'g_h0_lin')
            g_h0_re = tf.reshape(g_h0_lin, [-1, s_h16, s_w16, G_F_DIM * 8], name='g_h0_re')
            g_bn0 = ops.batch_norm(g_h0_re, name='g_bn0', train=False)
            h0 = tf.nn.relu(g_bn0)

            g_h1 = ops.deconv2d(h0, [self.sample_size, s_h8, s_w8, G_F_DIM * 4], name='g_h1')
            g_bn1 = ops.batch_norm(g_h1, name='g_bn1', train=False)
            h1 = tf.nn.relu(g_bn1)

            g_h2 = ops.deconv2d(h1, [self.sample_size, s_h4, s_w4, G_F_DIM * 2], name='g_h2')
            g_bn2 = ops.batch_norm(g_h2, name='g_bn2', train=False)
            h2 = tf.nn.relu(g_bn2)

            g_h3 = ops.deconv2d(h2, [self.sample_size, s_h2, s_w2, G_F_DIM * 1], name='g_h3')
            g_bn3 = ops.batch_norm(g_h3, name='g_bn3', train=False)
            h3 = tf.nn.relu(g_bn3)

            h4 = ops.deconv2d(h3, [self.sample_size, s_h, s_w, self.image_dims['channel']], name='g_h4')

            # 激活函数使用tanh
            return tf.nn.tanh(h4)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # 用5*5，stride为2的filter对输入进行卷积操作
            d_h0_conv = ops.conv2d(image, D_F_DIM, name='d_h0_conv')
            # 激活函数为leaky relu
            h0 = ops.lrelu(d_h0_conv)

            d_h1_conv = ops.conv2d(h0, D_F_DIM * 2, name='d_h1_conv')
            d_bn1 = ops.batch_norm(d_h1_conv, name='d_bn1', train=self.is_train)
            h1 = ops.lrelu(d_bn1)

            d_h2_conv = ops.conv2d(h1, D_F_DIM * 4, name='d_h2_conv')
            d_bn2 = ops.batch_norm(d_h2_conv, name='d_bn2', train=self.is_train)
            h2 = ops.lrelu(d_bn2)

            d_h3_conv = ops.conv2d(h2, D_F_DIM * 8, name='d_h3_conv')
            d_bn3 = ops.batch_norm(d_h3_conv, name='d_bn3', train=self.is_train)
            h3 = ops.lrelu(d_bn3)

            # 最后一层使用全连接
            h4 = ops.linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h4_lin')

            # return tf.nn.sigmoid(h4), h4
            return None, h4

    # def discriminator_for_sample(self, image):
    #     with tf.variable_scope("discriminator") as scope:
    #         scope.reuse_variables()
    #
    #         # 用5*5，stride为2的filter对输入进行卷积操作
    #         d_h0_conv = ops.conv2d(image, D_F_DIM, name='d_h0_conv')
    #         # 激活函数为leaky relu
    #         h0 = ops.lrelu(d_h0_conv)
    #
    #         d_h1_conv = ops.conv2d(h0, D_F_DIM * 2, name='d_h1_conv')
    #         d_bn1 = ops.batch_norm(d_h1_conv, name='d_bn1', train=False)
    #         h1 = ops.lrelu(d_bn1)
    #
    #         d_h2_conv = ops.conv2d(h1, D_F_DIM * 4, name='d_h2_conv')
    #         d_bn2 = ops.batch_norm(d_h2_conv, name='d_bn2', train=False)
    #         h2 = ops.lrelu(d_bn2)
    #
    #         d_h3_conv = ops.conv2d(h2, D_F_DIM * 8, name='d_h3_conv')
    #         d_bn3 = ops.batch_norm(d_h3_conv, name='d_bn3', train=False)
    #         h3 = ops.lrelu(d_bn3)
    #
    #         # 最后一层使用全连接
    #         h4 = ops.linear(tf.reshape(h3, [self.sample_size, -1]), 1, 'd_h4_lin')
    #
    #         return tf.nn.sigmoid(h4)

    def train(self, data):
        batch_manager = BatchManager(data, BATCH_SIZE)
        # adam梯度下降
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.RMSPropOptimizer(LEARNING_RATE) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(LEARNING_RATE) \
                .minimize(self.g_loss, var_list=self.g_vars)
        clip_d_op = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in self.d_vars]

        tf.global_variables_initializer().run()

        # 随机向量,用于一定时间的训练后，生成样本图片
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, Z_DIM))

        # 计数器，记录一共训练了多少次batch
        counter = 0
        start_time = time.time()
        # 读取检查点，即继续上次训练的参数继续训练
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            print("counter:{}".format(counter))
        else:
            print(" [!] Load failed...")

        for epoch in range(EPOCHS):
            # batch的数量
            batch_idxs = len(data) // BATCH_SIZE // CRITIC_NUM

            for idx in range(batch_idxs):
                # # 取出一个batch的图片
                # if is_file:
                #     batch_images_files = data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                #     batch_images = [get_image_from_file(path) for path in batch_images_files]
                # else:
                #     batch_images = data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

                if counter < 25 or counter % 500 == 0:
                    critic_num = 25
                else:
                    critic_num = CRITIC_NUM

                for _ in range(critic_num):
                    # 随机向量
                    batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                    # 训练D
                    self.sess.run([d_optim], feed_dict={
                        self.inputs: batch_manager.next_batch(),
                        self.z: batch_z,
                        self.is_train: True
                    })
                    self.sess.run(clip_d_op)

                # 训练G
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                self.sess.run([g_optim], feed_dict={
                    self.z: batch_z,
                    self.is_train: True,
                    self.inputs: batch_manager.cur_batch()
                })
                # 训练两次G
                # batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                self.sess.run([g_optim], feed_dict={self.z: batch_z,
                                                    self.is_train: True,
                                                    self.inputs: batch_manager.cur_batch()})

                err_d_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_train: False})
                err_d_real = self.d_loss_real.eval({self.inputs: batch_manager.cur_batch(), self.is_train: False})
                err_g = self.g_loss.eval({self.z: batch_z, self.is_train: False})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch + 1, EPOCHS, idx, batch_idxs,
                         time.time() - start_time, sigmoid(err_d_fake + err_d_real), sigmoid(err_g)))

                if np.mod(counter, 70) == 1:
                    # 生成样本，并未修改参数值
                    samples = self.sess.run(
                        [self.sample_img],
                        feed_dict={
                            self.z: sample_z,
                            self.is_train: False
                        },
                    )
                    save_images(samples, self.sample_dir, epoch + 1, counter, self.sample_size)
                    print("Sampling......")

                if np.mod(counter, 360) == 1:
                    self.save(self.checkpoint_dir, counter)

    # 用于保存模型到本地
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # 从本地读取
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def sample(self, dir, num):
        # 读取检查点，即继续上次训练的参数继续训练
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            print("counter:{}".format(counter))
        else:
            print(" [!] Load failed...")

        img = None
        count = int(math.ceil(num / BATCH_SIZE))
        for i in range(count):
            inputs_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM)).astype(np.float32)
            sampled_img = self.sess.run(
                [self.G],
                feed_dict={
                    self.z: inputs_z,
                    self.is_train: False
                },
            )
            sampled_img = np.squeeze(sampled_img)
            if i == 0:
                img = sampled_img
            else:
                img = np.append(img, sampled_img, axis=0)
            # img = np.squeeze(img)
            print(np.array(img).shape)
        img = img[:num]
        save_images(img, dir, sample_size=num)
        return img
        # loss = self.sess.run(
        #     [self.temp_D], feed_dict={
        #         self.temp_img: sampled_img,
        #         self.is_train: False}
        # )
        #
        # print(loss)


def sigmoid(x):
    sigmoid_scores = 1 / float(1 + math.exp(-x))
    return sigmoid_scores
