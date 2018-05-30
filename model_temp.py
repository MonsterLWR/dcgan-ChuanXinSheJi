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


class DCGAN():
    def __init__(self, sess, height=64, width=64, channel=3,
                 checkpoint_dir=None, sample_dir=None, sample_size=64, lam=0.1):
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
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        # 用于产生图片，与generator共享参数
        self.sample_img = self.sampler(self.z)
        # D_ 代表判别器输入生成图片产生的输出
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # sample用
        # self.temp_img = tf.placeholder(
        #     tf.float32, [None,
        #                  self.image_dims['height'],
        #                  self.image_dims['width'],
        #                  self.image_dims['channel']
        #                  ], name='temp_img')
        # self.temp_D = self.discriminator(self.temp_img, reuse=True)

        # 判别其对真实图片的损失函数
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # 判别其对生成图片的损失函数
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        # 生成器的损失函数
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # 判别器的损失函数
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # Returns all variables created with trainable=True
        t_vars = tf.trainable_variables()

        # 分别得到生成器和判别其参数，用于使用optimizer时限制可更新的参数
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # 用于保存参数到硬盘的对象
        self.saver = tf.train.Saver()

        # Completion.
        self.lam = lam
        self.mask = tf.placeholder(tf.float32, [height, width, channel], name='mask')
        # tf.contrib.layers.flatten()第二个参数作用不明
        # [None,1]
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.inputs))), axis=1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        # 取导数
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

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

            return tf.nn.sigmoid(h4), h4

    def train(self, data, is_file=False):
        # adam梯度下降
        d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
            .minimize(self.g_loss, var_list=self.g_vars)

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
            # 打乱data
            np.random.shuffle(data)
            # batch的数量
            batch_idxs = len(data) // BATCH_SIZE

            for idx in range(batch_idxs):
                # 取出一个batch的图片
                if is_file:
                    batch_images_files = data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                    batch_images = [get_image_from_file(path) for path in batch_images_files]
                else:
                    batch_images = data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

                # 随机向量
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                # 训练D
                self.sess.run([d_optim], feed_dict={
                    self.inputs: batch_images,
                    self.z: batch_z,
                    self.is_train: True
                })

                # 训练G
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                self.sess.run([g_optim], feed_dict={self.z: batch_z, self.is_train: True})
                # 训练两次G
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                self.sess.run([g_optim], feed_dict={self.z: batch_z, self.is_train: True})
                # # 训练三次G
                # batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                # self.sess.run([g_optim], feed_dict={self.z: batch_z})

                err_d_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_train: False})
                err_d_real = self.d_loss_real.eval({self.inputs: batch_images, self.is_train: False})
                err_g = self.g_loss.eval({self.z: batch_z, self.is_train: False})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch + 1, EPOCHS, idx, batch_idxs,
                         time.time() - start_time, err_d_fake + err_d_real, err_g))

                if np.mod(counter, 300) == 1:
                    # 生成样本，并未修改参数值
                    samples = self.sess.run(
                        [self.sample_img],
                        feed_dict={
                            self.z: sample_z,
                            self.is_train: False
                        },
                    )
                    save_images(samples, self.sample_dir, epoch + 1, idx, self.sample_size)
                    print("Sampling......")

                if np.mod(counter, 1800) == 1:
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

    def sample(self, sample_num):
        # 读取检查点，即继续上次训练的参数继续训练
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            print("counter:{}".format(counter))
        else:
            print(" [!] Load failed...")

        inputs_z = np.random.uniform(-1, 1, size=(sample_num, Z_DIM)).astype(np.float32)

        sampled_img = self.sess.run(
            [self.sample_img],
            feed_dict={
                self.z: inputs_z,
                self.is_train: False
            },
        )
        sampled_img = np.squeeze(sampled_img)

        return sampled_img
        # loss = self.sess.run(
        #     [self.sample_D], feed_dict={
        #         self.inputs: sampled_img,
        #         self.is_train: False}
        # )
        #
        # print(loss)

    def load_model(self):
        # 读取检查点，即继续上次训练的参数继续训练
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            print("counter:{}".format(counter))
        else:
            print(" [!] Load failed...")

        return could_load, checkpoint_counter

    def complete(self, imgs, mask_type='center', center_scale=0.5,
                 out_dir=None, num_iter=500, out_interval=100):
        img_shape = imgs[0].shape
        img_size = img_shape[0:2]
        nImgs = len(imgs)

        tf.global_variables_initializer().run()

        isLoaded, _ = self.load_model()
        assert (isLoaded)

        # 一次补全一个batch_size的图片
        batch_idxs = int(np.ceil(nImgs / BATCH_SIZE))
        # lowres_mask = np.zeros(self.lowres_shape)
        if mask_type == 'random':
            fraction_masked = 0.2
            mask = np.ones(img_shape)
            mask[np.random.random(img_shape[:2]) < fraction_masked] = 0.0
        elif mask_type == 'center':
            assert (center_scale <= 0.5)
            mask = np.ones(img_shape)
            l = int(img_size * center_scale)
            u = int(img_size * (1.0 - center_scale))
            mask[l:u, l:u, :] = 0.0
        elif mask_type == 'left':
            mask = np.ones(img_shape)
            c = img_shape[0] // 2
            mask[:, :c, :] = 0.0
        elif mask_type == 'full':
            mask = np.ones(img_shape)
        elif mask_type == 'grid':
            mask = np.zeros(img_shape)
            mask[::4, ::4, :] = 1.0
        # elif maskType == 'lowres':
        #     lowres_mask = np.ones(self.lowres_shape)
        #     mask = np.zeros(self.image_shape)
        else:
            assert (False, 'no such musk type!')

        for idx in range(0, batch_idxs):
            l = idx * BATCH_SIZE
            # 本次batch的最后一张图片的索引
            u = min((idx + 1) * BATCH_SIZE, nImgs)
            # 得到图片数据
            batch_images = np.array(imgs[l:u]).astype(np.float32)

            # 本次idx实际图片数目
            batchSz = u - l
            # 图片数目小于一个batch
            if batchSz < BATCH_SIZE:
                # 添加值全为0的图片将图片数量补全为batch_size
                # 便于生成大图片
                padSz = ((0, int(BATCH_SIZE - batchSz)), (0, 0), (0, 0), (0, 0))
                # constant 默认为零
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))
            # 用于手动构建AdamOptimizer
            m = 0
            v = 0

            # 将生成图片合并为一张大的图片
            # nRow，nCol代表大图片一行和一列可以放多少小图片
            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            # 将该batch的图片合并为一张大图
            merge_and_save(batch_images, [nRows, nCols], out_dir, 'before', idx)
            masked_images = np.multiply(batch_images, mask)
            merge_and_save(batch_images, [nRows, nCols], out_dir, 'masked', idx)
            # # 如果lowres_mask存在为1的值，生成lowres的图片
            # if lowres_mask.any():
            #     lowres_images = np.reshape(batch_images, [self.batch_size, self.lowres_size, self.lowres,
            #                                               self.lowres_size, self.lowres, self.c_dim]).mean(4).mean(2)
            #     lowres_images = np.multiply(lowres_images, lowres_mask)
            #     lowres_images = np.repeat(np.repeat(lowres_images, self.lowres, 1), self.lowres, 2)
            #     save_images(lowres_images[:batchSz, :, :, :], [nRows, nCols],
            #                 os.path.join(config.outDir, 'lowres.png'))
            for img in range(batchSz):
                with open(os.path.join(out_dir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                    f.write('iter loss ' +
                            ' '.join(['z{}'.format(zi) for zi in range(Z_DIM)]) +
                            '\n')

            # 开始训练z
            for i in range(num_iter):
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.lowres_mask: lowres_mask,
                    self.images: batch_images,
                    self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G]
                loss, g, G_imgs, lowres_G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(out_dir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img + 1])

                if i % out_interval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    # imgName = os.path.join(outDir, 'hats_imgs/{:04d}.png'.format(i))
                    # nRows = np.ceil(batchSz / 8)
                    # nCols = min(8, batchSz)
                    merge_and_save(G_imgs[:batchSz, :, :, :], [nRows, nCols],
                                   out_dir, 'generated{}'.format(i), idx)
                    # if lowres_mask.any():
                    #     imgName = imgName[:-4] + '.lowres.png'
                    #     save_images(np.repeat(np.repeat(lowres_G_imgs[:batchSz, :, :, :],
                    #                                     self.lowres, 1), self.lowres, 2),
                    #                 [nRows, nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
                    completed = masked_images + inv_masked_hat_images
                    # imgName = os.path.join(config.outDir,
                    #                        'completed/{:04d}.png'.format(i))
                    # save_images(completed[:batchSz, :, :, :], [nRows, nCols], imgName)
                    merge_and_save(completed[:batchSz, :, :, :], [nRows, nCols],
                                   out_dir, 'completed{}'.format(i), idx)

                # if config.approach == 'adam':
                # Optimize single completion with Adam
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - config.beta1 ** (i + 1))
                v_hat = v / (1 - config.beta2 ** (i + 1))
                zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                zhats = np.clip(zhats, -1, 1)
                #
                # elif config.approach == 'hmc':
                #     # Sample example completions with HMC (not in paper)
                #     zhats_old = np.copy(zhats)
                #     loss_old = np.copy(loss)
                #     v = np.random.randn(self.batch_size, self.z_dim)
                #     v_old = np.copy(v)
                #
                #     for steps in range(config.hmcL):
                #         v -= config.hmcEps / 2 * config.hmcBeta * g[0]
                #         zhats += config.hmcEps * v
                #         np.copyto(zhats, np.clip(zhats, -1, 1))
                #         loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                #         v -= config.hmcEps / 2 * config.hmcBeta * g[0]
                #
                #     for img in range(batchSz):
                #         logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img] ** 2) / 2
                #         logprob = config.hmcBeta * loss[img] + np.sum(v[img] ** 2) / 2
                #         accept = np.exp(logprob_old - logprob)
                #         if accept < 1 and np.random.uniform() > accept:
                #             np.copyto(zhats[img], zhats_old[img])
                #
                #     config.hmcBeta *= config.hmcAnneal
                #
                # else:
                #     assert (False)
