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

def create_generator_encoder(ni, a):
	W_init = tf.random_normal_initializer(0, 0.02)
	g_init = tf.random_normal_initializer(1.0, 0.02)

	nn = Conv2dLayer(ni, a.ngf, (4, 4), (2, 2), act=lambda x:tl.act.lrelu(x, 0.2), padding='SAME', W_init=W_init)
	nn = Conv2d(a.ngf*2, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)
	nn = Conv2d(a.ngf*4, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)
	nn = Conv2d(a.ngf*8, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn)

	nn2 = Conv2d(a.ngf*8, (4, 4), (2, 2), padding='SAME', W_init=W_init)(nn)
	nn2 = BatchNorm(act=lambda x:tl.act.lrelu(x, 0.2), gamma_init=g_init)(nn2)
	nn3 = Reshape([-1, 16*16*8*a.ngf])(nn)
	nn3 = Dense(8)(nn3)

	sR=nn2
	eR=nn3
	return sR, eR