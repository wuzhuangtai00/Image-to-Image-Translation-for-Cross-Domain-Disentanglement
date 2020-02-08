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

def create_generator_decoder_exclusive(eR, out, a):
	W_init = tf.random_normal_initializer(0, 0.02)
	g_init = tf.random_normal_initializer(1.0, 0.02)

	n = eR.shape[0]
	dim = eR.shape[-1]
	image_size = 8
	z = tf.reshape(eR, [n, 1, 1, dim])
	z = tf.tile(z, [1, image_size, image_size, 1])
	ni = z

	g = tf.get_default_graph()
	with g.gradient_override_map({"Identity": "ReverseGrad"}):
		nn = tf.identity(ni)

	nn = tf.nn.relu(nn)
	nn = DeConv2d(a.ngf * 8, (4, 4), (2, 2), padding='SAME', W_init=W_init)(ni)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)
	nn = Dropout(keep=0.5)(nn)
	nn = tf.nn.relu(nn)
	nn = DeConv2d(a.ngf * 4, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)
	nn = Dropout(keep=0.5)(nn)
	nn = tf.nn.relu(nn)
	nn = DeConv2d(a.ngf * 2, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)
	nn = tf.nn.relu(nn)
	nn = DeConv2d(a.ngf, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)
	nn = tf.nn.relu(nn)
	nn = DeConv2d(out, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = tf.tanh(nn)

	return nn
