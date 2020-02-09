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

def BatchNorm(ni, pre, num):
	g_init = tf.random_normal_initializer(1.0, 0.02)
	inputs = ni.outputs
	tmp = tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=g_init)
	tmp = lrelu(tmp, 0.2)
	return InputLayer(tmp, name=pre+'Batch'+num)

def create_generator_encoder(ni, a, pre):
	W_init = tf.random_normal_initializer(0, 0.02)
	g_init = tf.random_normal_initializer(1.0, 0.02)

	print(ni.outputs)

	nn = Conv2d(ni, a.ngf, filter_size=(4, 4), strides=(2, 2), act=lambda x:tl.act.lrelu(x, 0.2), padding='SAME', W_init=W_init, name=pre+'_Conv2d1')
	nn = Conv2d(nn, a.ngf*2, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_Conv2d2')
	nn = Batch(nn, pre, '1')
#	nn = BatchNormLayer(nn, act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init, name=pre+'_Batch1')
	nn = Conv2d(nn, a.ngf*4, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_Conv2d3')
	nn = Batch(nn, pre, '2')
#	nn = BatchNormLayer(nn, act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init, name=pre+'_Batch2')
	nn = Conv2d(nn, a.ngf*8, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_Conv2d4')
	nn = Batch(nn, pre, '3')
#	nn = BatchNormLayer(nn, act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init, name=pre+'_Batch3')

	nn2 = Conv2d(nn, a.ngf*8, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_Conv2d5')
	nn = Batch(nn, pre, '4')
#	nn2 = BatchNormLayer(nn2, act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init, name=pre+'_Batch4')
	nn3 = ReshapeLayer(nn, [-1, 16*16*8*a.ngf], name=pre+'_Reshape1')
	nn3 = DenseLayer(nn3, 8, name=pre+'_Dense1')

	sR=nn2
	eR=nn3
	return sR, eR
