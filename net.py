import os
from keras.layers import Dense, Conv2DTranspose, Reshape, UpSampling2D, Conv2D, LeakyReLU, Flatten, Activation, BatchNormalization
from keras.models import Sequential
import numpy as np
import tensorflow as tf

class Net(object):
    def __init__(self, z_dim=128, dim=64, gen_model=None, dis_model=None):
        if gen_model is None:
            gen_model = Sequential()
            gen_model.add(Dense(12*12*512, activation='relu', input_dim=z_dim))
            gen_model.add(Activation('relu'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Reshape([512, 12, 12]))
            gen_model.add(Conv2DTranspose(256, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))
            gen_model.add(Conv2DTranspose(128, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))
            gen_model.add(Conv2DTranspose(64, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))
            gen_model.add(Conv2DTranspose(3, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(Activation('tanh'))
        self.generator = gen_model

        if dis_model is None:
            dis_model = Sequential()
            dis_model.add(Conv2D(dim, 5, strides=2, padding='same', data_format='channels_first', input_shape=[3, 64, 64]))
            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Conv2D(dim*2, 5, strides=2))
            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Flatten())
            dis_model.add(Dense(256))
            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Dense(1))
        self.discriminator = dis_model

    def save_models(self, name, save_dir='save'):
        self.generator.save(os.path.join(save_dir, "generator_{}.h5".format(name)))
        self.discriminator.save(os.path.join(save_dir, "discriminator_{}.h5".format(name)))

if __name__ == '__main__':
    net = Net()

