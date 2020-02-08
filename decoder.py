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

def create_generator_decoder(sR, eR, out, a, pre, noise=True):
	W_init = tf.random_normal_initializer(0, 0.02)
	g_init = tf.random_normal_initializer(1.0, 0.02)

	n = sR.outputs.shape[0]
	dim = eR.outputs.shape[-1]
	sz = sR.outputs.shape[1]
	z = tf.reshape(eR.outputs, [n, 1, 1, dim])
	z = tf.tile(z, [1, sz, sz, 1])
	tmp = tf.concat([sR.outputs, z],axis=3)

	if a.mode == "train":
		if noise:
			inoise = tf.random_normal(tmp.shape, mean=0.0, stddev=a.noise)
			tmp += inoise

#	ni = tf.nn.relu(ni)
	ni = InputLayer(tmp, name=pre+'_input')
	nn = DeConv2d(ni, a.ngf * 8, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv2d1')
	print('[TL]', end = ''),
	print(nn.outputs)
	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch1')
	print('[TL]', end = ''),
	print(nn.outputs)
	if a.mode == "train":
		nn = DropoutLayer(nn, keep=0.5, name=pre+'_Dropout1')
		print('[TL]', end = ''),
		print(nn.outputs)
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, a.ngf * 4, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv2d2')
	print('[TL]', end = ''),
	print(nn.outputs)
	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch2')
	if a.mode == "train":
		nn = DropoutLayer(nn, keep=0.5, name=pre+'_Dropout2')
		print('[TL]', end = ''),
		print(nn.outputs)
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, a.ngf * 2, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv2d3')
	print('[TL]', end = ''),
	print(nn.outputs)
	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch3')
	print('[TL]', end = ''),
	print(nn.outputs)
	#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, a.ngf, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv2d4')
	print('[TL]', end = ''),
	print(nn.outputs)
	nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, name=pre+'_Batch4')
	print('[TL]', end = ''),
	print(nn.outputs)
#	nn = tf.nn.relu(nn)
	nn = DeConv2d(nn, out, filter_size=(4, 4), strides=(2, 2), padding='SAME', W_init=W_init, name=pre+'_DeConv2d5')
	print('[TL]', end = ''),
	print(nn.outputs)
	
	return InputLayer(tf.tanh(nn.outputs), name=pre+'_ans')
