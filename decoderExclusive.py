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

@tf.RegisterGradient("ReverseGrad")
def _reverse_grad(unused_op, grad):
	return -1.0*grad

def Batch(ni, pre, num):
	g_init = tf.random_normal_initializer(1.0, 0.02)
	tmp = tf.layers.batch_normalization(ni.outputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=g_init)
	tmp = tf.nn.relu(tmp)
	return InputLayer(tmp, name=pre+'Batch'+num)

def create_generator_decoder_exclusive(eR, out, a, pre):
	W_init = tf.random_normal_initializer(0, 0.02)
	g_init = tf.random_normal_initializer(1.0, 0.02)

	n = eR.outputs.shape[0]
	dim = eR.outputs.shape[-1]
	image_size = 8
	z = tf.reshape(eR.outputs, [n, 1, 1, dim])
	z = tf.tile(z, [1, image_size, image_size, 1])
	ni = z

	g = tf.get_default_graph()
	with g.gradient_override_map({"Identity": "ReverseGrad"}):
		ni = tf.identity(ni)

	ni = tf.nn.relu(ni)
	nn = InputLayer(ni, name=pre+'_Input')
	nn = DeConv2d(nn, a.ngf * 8, (4, 4), (2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv1')
	nn = Batch(nn, pre, '1')
#	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch1')
	nn = DropoutLayer(nn, keep=0.5, name=pre+'_Drop1', is_fix = True)
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, a.ngf * 4, (4, 4), (2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv2')
	nn = Batch(nn, pre, '2')
#	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch2')
	nn = DropoutLayer(nn, keep=0.5, name=pre+'_Drop2', is_fix = True)
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, a.ngf * 2, (4, 4), (2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv3')
	nn = Batch(nn, pre, '3')
#	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch3')
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, a.ngf, (4, 4), (2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv4')
	nn = Batch(nn, pre, '4')
#	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch4')
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, out, (4, 4), (2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv5')

	return InputLayer(tf.tanh(nn.outputs), name=pre+'_ans')
