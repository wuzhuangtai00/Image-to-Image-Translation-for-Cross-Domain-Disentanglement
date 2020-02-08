from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def work(ni, out, pre, num):
	W_init = tf.random_normal_initializer(0, 0.02)
	tmp = tf.pad(ni.outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
	tmp2 = InputLayer(tmp, name=pre+'_Input'+num)
	nn = Conv2d(tmp2, out, (4, 4), (2, 2), act=lambda x:tl.act.lrelu(x, 0.2), padding='VALID', W_init=W_init, name=pre+'Conv2d'+num)
	return nn

def create_discriminator(input, target, a, pre):
	print(input.outputs)
	print(target.outputs)
	ni = InputLayer(tf.concat([input.outputs, target.outputs], axis=3), name=pre+'_Input')

	nn = work(ni, a.ndf, pre, '1')
	nn = work(nn, a.ndf*2, pre, '2')
	nn = work(nn, a.ndf*4, pre, '3')
	nn = work(nn, a.ndf*8, pre, '4')
	nn = ReshapeLayer(nn, [-1, 4*4*4*a.ndf], name=pre+'_Reshape1')
	nn = DenseLayer(nn, 1, name=pre+'_Dense')
	nn = ReshapeLayer(nn, [-1], name=pre+'_Reshape2')

	return nn
