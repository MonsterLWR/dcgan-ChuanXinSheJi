import tensorflow as tf
import time
from utils import *
import oprations as ops

# 超参数设定
EPOCHS = 1
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

        self.loaded = False
        self.counter = 0

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
            g_bn0 = tf.layers.batch_normalization(g_h0_re, name='g_bn0', training=self.is_train)
            h0 = tf.nn.relu(g_bn0)

            g_h1 = ops.deconv2d(h0, [BATCH_SIZE, s_h8, s_w8, G_F_DIM * 4], name='g_h1')
            g_bn1 = tf.layers.batch_normalization(g_h1, name='g_bn1', training=self.is_train)
            h1 = tf.nn.relu(g_bn1)

            g_h2 = ops.deconv2d(h1, [BATCH_SIZE, s_h4, s_w4, G_F_DIM * 2], name='g_h2')
            g_bn2 = tf.layers.batch_normalization(g_h2, name='g_bn2', training=self.is_train)
            h2 = tf.nn.relu(g_bn2)

            g_h3 = ops.deconv2d(h2, [BATCH_SIZE, s_h2, s_w2, G_F_DIM * 1], name='g_h3')
            g_bn3 = tf.layers.batch_normalization(g_h3, name='g_bn3', training=self.is_train)
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
            g_bn0 = tf.layers.batch_normalization(g_h0_re, name='g_bn0', training=False)
            h0 = tf.nn.relu(g_bn0)

            g_h1 = ops.deconv2d(h0, [self.sample_size, s_h8, s_w8, G_F_DIM * 4], name='g_h1')
            g_bn1 = tf.layers.batch_normalization(g_h1, name='g_bn1', training=False)
            h1 = tf.nn.relu(g_bn1)

            g_h2 = ops.deconv2d(h1, [self.sample_size, s_h4, s_w4, G_F_DIM * 2], name='g_h2')
            g_bn2 = tf.layers.batch_normalization(g_h2, name='g_bn2', training=False)
            h2 = tf.nn.relu(g_bn2)

            g_h3 = ops.deconv2d(h2, [self.sample_size, s_h2, s_w2, G_F_DIM * 1], name='g_h3')
            g_bn3 = tf.layers.batch_normalization(g_h3, name='g_bn3', training=False)
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
            d_bn1 = tf.layers.batch_normalization(d_h1_conv, name='d_bn1', training=self.is_train)
            h1 = ops.lrelu(d_bn1)

            d_h2_conv = ops.conv2d(h1, D_F_DIM * 4, name='d_h2_conv')
            d_bn2 = tf.layers.batch_normalization(d_h2_conv, name='d_bn2', training=self.is_train)
            h2 = ops.lrelu(d_bn2)

            d_h3_conv = ops.conv2d(h2, D_F_DIM * 8, name='d_h3_conv')
            d_bn3 = tf.layers.batch_normalization(d_h3_conv, name='d_bn3', training=self.is_train)
            h3 = ops.lrelu(d_bn3)

            # 最后一层使用全连接
            h4 = ops.linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def load_model(self):
        self.loaded = True
        # 读取检查点，即继续上次训练的参数继续训练
        # 返回已经训练的batch_counter
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
            print("counter:{}".format(checkpoint_counter))
            return checkpoint_counter
        else:
            print(" [!] Load failed...")
            return 0

    def train(self, top_data_dir):
        print('start training...')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # adam梯度下降
            d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
                .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.counter = self.load_model()

        # 计数器，记录一共训练了多少次batch
        start_time = time.time()

        # 随机向量,用于一定时间的训练后，生成样本图片
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, Z_DIM))

        data_dirs = glob.glob(os.path.join(top_data_dir, '*'))

        for epoch in range(EPOCHS):
            for data_dir in data_dirs:
                print('start training img in {}'.format(data_dir))
                data_files = glob.glob(os.path.join(data_dir, '*'))
                # batch的数量
                batch_idxs = len(data_files) // BATCH_SIZE
                batch_maneger = BatchManager(data_files, BATCH_SIZE)

                for idx in range(batch_idxs):
                    # 取出一个batch的图片
                    # if is_file:
                    #     batch_images_files = data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                    #     batch_images = [get_image_from_file(path) for path in batch_images_files]
                    # else:
                    # batch_images = data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                    batch_images = batch_maneger.next_batch()

                    # 随机向量
                    batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                    # 训练D
                    self.sess.run([d_optim], feed_dict={
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.is_train: True
                    })

                    # 训练G
                    # batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                    self.sess.run([g_optim], feed_dict={
                        self.z: batch_z,
                        self.is_train: True,
                        self.inputs: batch_images
                    })
                    # 训练两次G
                    # batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                    self.sess.run([g_optim], feed_dict={
                        self.z: batch_z,
                        self.is_train: True,
                        self.inputs: batch_images})

                    err_d_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_train: False})
                    err_d_real = self.d_loss_real.eval({self.inputs: batch_images, self.is_train: False})
                    err_g = self.g_loss.eval({self.z: batch_z, self.is_train: False})

                    self.counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch + 1, EPOCHS, idx, batch_idxs,
                             time.time() - start_time, err_d_fake + err_d_real, err_g))

                    if np.mod(self.counter, 150) == 1 and self.counter != 1:
                        # 生成样本，并未修改参数值
                        samples = self.sess.run(
                            [self.sample_img],
                            feed_dict={
                                self.z: sample_z,
                                self.is_train: False
                            },
                        )
                        save_images(samples, self.sample_dir, epoch + 1, self.counter, self.sample_size)
                        print("Sampling......")

                    if np.mod(self.counter, 451) == 1:
                        self.save(self.checkpoint_dir, self.counter)

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
        save_images(img, dir, sample_num=num)
        return img

    def complete(self, imgs, mask_type='center', center_scale=0.3,
                 out_dir=None, num_iter=1501, out_interval=100):
        img_shape = imgs[0].shape
        img_size = img_shape[1]
        nImgs = len(imgs)

        os.makedirs(os.path.join(out_dir, 'logs'))

        tf.global_variables_initializer().run()

        self.load(self.checkpoint_dir)

        # 一次补全一个batch_size的图片
        batch_idxs = int(np.ceil(nImgs / BATCH_SIZE))
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
            lr = 0.001
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-08

            # 将生成图片合并为一张大的图片
            # nRow，nCol代表大图片一行和一列可以放多少小图片
            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            # 将该batch的图片合并为一张大图
            merge_and_save(batch_images, [nRows, nCols], out_dir, 'before', idx)
            masked_images = np.multiply(batch_images, mask)
            merge_and_save(masked_images, [nRows, nCols], out_dir, 'masked', idx)
            # for img in range(batchSz):
            #     with open(os.path.join(out_dir, 'logs\\hats_{:02d}.log'.format(img)), 'a') as f:
            #         f.write('iter loss ' +
            #                 ' '.join(['z{}'.format(zi) for zi in range(Z_DIM)]) +
            #                 '\n')

            # 开始训练z
            for i in range(num_iter):
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.inputs: batch_images,
                    self.is_train: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, grad_loss, g_imgs = self.sess.run(run, feed_dict=fd)
                if i % 10 == 0:
                    print('iteration:{},loss:{}'.format(i, np.mean(loss[0:batchSz])))

                # for img in range(batchSz):
                #     with open(os.path.join(out_dir, 'logs\\hats_{:02d}.log'.format(img)), 'ab') as f:
                #         f.write('{} {} '.format(i, loss[img]).encode())
                #         np.savetxt(f, zhats[img:img + 1])

                if i % out_interval == 0:
                    # print(i, np.mean(loss[0:batchSz]))
                    merge_and_save(g_imgs[:batchSz, :, :, :], [nRows, nCols],
                                   out_dir, 'generated{}'.format(i), idx)

                    inv_masked_hat_images = np.multiply(g_imgs, 1.0 - mask)
                    completed = masked_images + inv_masked_hat_images
                    merge_and_save(completed[:batchSz, :, :, :], [nRows, nCols],
                                   out_dir, 'completed{}'.format(i), idx)

                # if config.approach == 'adam':
                # Optimize single completion with Adam
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * grad_loss[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(grad_loss[0], grad_loss[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))
                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + epsilon))
                zhats = np.clip(zhats, -1, 1)
