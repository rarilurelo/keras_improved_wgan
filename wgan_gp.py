from keras import backend as K
import numpy as np
import tensorflow as tf

class WganGP(object):
    def __init__(self, net, data_feeder, sess, batch_size=64, dim=64, z_dim=128, lambda_gp=10, dis_lr=1e-4, gen_lr=1e-4, n_critic=5):
        self.net = net
        self.data_feeder = data_feeder
        self.sess = sess
        self.batch_size = batch_size
        self.dim = dim
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.dis_lr = dis_lr
        self.gen_lr = gen_lr
        self.n_critic = n_critic
        self.built = False

    def build(self):
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.real_image = tf.placeholder(tf.float32, shape=[None, 3, 64, 64])
        self.fake_image = self.net.generator(self.z)
        self.dis_fake = self.net.discriminator(self.fake_image)
        self.dis_real = self.net.discriminator(self.real_image)
        self.gen_loss = -K.mean(self.dis_fake)
        self.dis_loss = K.mean(self.dis_fake) - K.mean(self.dis_real)
        alpha = K.random_uniform(shape=[K.shape(self.z)[0], 1, 1, 1])
        diff = self.fake_image - self.real_image
        interp = self.real_image + alpha * diff
        gradients = K.gradients(self.net.discriminator(interp), [interp])[0]
        gp = K.mean(K.square(K.sqrt(K.sum(K.square(gradients), axis=1))-1))
        self.dis_loss += self.lambda_gp * gp

        self.dis_updater = tf.train.AdamOptimizer(learning_rate=self.dis_lr).minimize(self.dis_loss, var_list=self.net.discriminator.trainable_weights)
        self.gen_updater = tf.train.AdamOptimizer(learning_rate=self.gen_lr).minimize(self.gen_loss, var_list=self.net.generator.trainable_weights)
        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def train(self, epoch):
        if not self.built:
            self.build()
        for i in range(epoch):
            for _ in range(self.n_critic):
                images = self.data_feeder.fetch_data()
                feed_in = {self.z: np.random.normal(size=[self.batch_size, self.z_dim]),
                        self.real_image: images}
                self.sess.run(self.dis_updater, feed_in)
            self.sess.run(self.gen_updater, {self.z: np.random.normal(size=[self.batch_size, self.z_dim])})
            print("epoch: {}, gen_loss: {}, dis_loss{}".format(i, *self.sess.run([self.gen_loss, self.dis_loss], feed_in)))

    def generate_image(self, z, names, concat, save_dir='save'):
        if not self.built:
            self.build()
        self.data_feeder.save_images(self.sess.run(self.fake_image, {self.z: z}), names, concat, save_dir)

    def save_models(self, name, save_dir='save'):
        self.net.save_models(name, save_dir)

