import argparse

import keras.backend as K
import numpy as np
import tensorflow as tf

from net import Net
from preprocess import DataFeeder
from wgan_gp import WganGP

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', type=str, default='data')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=64)
args = parser.parse_args()
load_dir = args.load_dir
batch_size = args.batch_size
size = (args.image_size, args.image_size)

net = Net()
data_feeder = DataFeeder(load_dir=load_dir, batch_size=batch_size, size=size)
sess = tf.Session()
wgan_gp = WganGP(net, data_feeder, sess, batch_size, size[0])

wgan_gp.train(100)

wgan_gp.generate_image(np.random.normal(size=[10, 128]), 'tekitou', concat=True)

