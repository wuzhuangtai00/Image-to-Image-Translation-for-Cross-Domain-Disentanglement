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

## import tensorflow.compat.v1 as tf
## tf.disable_v2_behavior()
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import *

from encoder import *
from decoder import *
from decoderExclusive import *
from discriminatorWGANGP import *


LAMBDA = 10

Model = collections.namedtuple("Model", "outputsX2Y, outputsY2X,\
                               outputsX2Yp, outputsY2Xp,\
                               outputs_exclusiveX2Y,outputs_exclusiveY2X,\
                               discrim_exclusiveX2Y_loss,discrim_exclusiveY2X_loss,\
                               auto_outputX, auto_outputY\
                               predict_realX2Y, predict_realY2X,\
                               predict_fakeX2Y, predict_fakeY2X,\
                               sR_X2Y,sR_Y2X,\
                               eR_X2Y,eR_Y2X,\
                               discrimX2Y_loss, discrimY2X_loss,\
                               genX2Y_loss, genY2X_loss,\
                               gen_exclusiveX2Y_loss,gen_exclusiveY2X_loss\
                               autoencoderX_loss, autoencoderY_loss,\
                               feat_recon_loss,code_recon_loss,\
                               code_sR_X2Y_recon_loss,code_sR_Y2X_recon_loss,\
                               code_eR_X2Y_recon_loss,code_eR_Y2X_recon_loss,\
                               im_swapped_Y,sel_auto_Y\
                               im_swapped_X,sel_auto_X\
                               train")

def create_model(inputsX, inputsY, a):
    IMAGE_SIZE = 256
    OUTPUT_DIM = IMAGE_SIZE*IMAGE_SIZE*3

    targetsX = inputsY
    targetsY = inputsX

    print(inputsX, inputsY)
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')

##    niX = Input([None, IMAGE_SIZE, IMAGE_SIZE, 3])
##    niY = Input([None, IMAGE_SIZE, IMAGE_SIZE, 3])
    niX = InputLayer(inputsX, name='inputsX')
    niY = InputLayer(inputsY, name='inputsY')

    taX = InputLayer(inputsY, name='targetX')
    taY = InputLayer(inputsX, name='targetY')
    
    ######### IMAGE_TRANSLATORS
    with tf.variable_scope("generatorX2Y_encoder"):
        sR_X2Y, eR_X2Y = create_generator_encoder(niX, a, 'generatorX2Y_encoder_X2Y')
    with tf.variable_scope("generatorY2X_encoder"):
        sR_Y2X, eR_Y2X = create_generator_encoder(niY, a, 'generatorX2Y_encoder_Y2X')

    # layer has no shape
    z = InputLayer(tf.random_normal(eR_X2Y.outputs.shape), name='__z')
    z2 = InputLayer(tf.random_normal(eR_X2Y.outputs.shape), name='__z2')

    with tf.name_scope("generatorX2Y_decoder_noise"):
        with tf.variable_scope("generatorX2Y_decoder"):
            out = int(targetsX.get_shape()[-1])
            outputsX2Y = create_generator_decoder(sR_X2Y, z, out, a, 'generatorX2Y_decoder_noise_generatorX2Y_decoder')
            print(outputsX2Y.outputs)
        with tf.variable_scope("generatorX2Y_decoder", reuse=tf.AUTO_REUSE):
            outputsX2Yp = create_generator_decoder(sR_X2Y, z2, out, a, 'generatorX2Y_decoder_noise_generatorX2Y_decoder_reuse')
    with tf.name_scope("generatorX2Y_reconstructor"):
        with tf.variable_scope("generatorY2X_encoder", reuse=tf.AUTO_REUSE):
            sR_X2Y_recon, eR_X2Y_recon = create_generator_encoder(outputsX2Y, a, 'generatorX2Y_reconstructor_generatorY2X_encoder_reuse')

    with tf.name_scope("generatorY2X_decoder_noise"):
        with tf.variable_scope("generatorY2X_decoder"):
            out = int(targetsY.get_shape()[-1])
            outputsY2X = create_generator_decoder(sR_Y2X, z, out, a, 'generatorY2X_decoder_noise_generatorY2X_decoder')
        with tf.variable_scope("generatorY2X_decoder",reuse=tf.AUTO_REUSE):
            outputsY2Xp = create_generator_decoder(sR_Y2X, z2, out, a, 'generatorY2X_decoder_noise_generatorY2X_decoder_reuse')
    with tf.name_scope("generatorY2X_reconstructor"):
        with tf.variable_scope("generatorX2Y_encoder", reuse=tf.AUTO_REUSE):
            sR_Y2X_recon, eR_Y2X_recon = create_generator_encoder(outputsY2X, a, 'generatorY2X_reconstructor_generatorX2Y_encoder_reuse')

    ######### CROSS-DOMAIN AUTOENCODERS
    with tf.name_scope("autoencoderX"):
        # Use here decoder Y2X but with shared input from X2Y encoder
        with tf.variable_scope("generatorY2X_decoder", reuse=tf.AUTO_REUSE):
            out = int(inputsX.get_shape()[-1])
            auto_outputX = create_generator_decoder(sR_Y2X, eR_X2Y, out, a, 'autoencoderX_generatorY2X_decoder_reuse')
    with tf.name_scope("autoencoderY"):
        # Use here decoder X2Y but with input from Y2X encoder
        with tf.variable_scope("generatorX2Y_decoder", reuse=tf.AUTO_REUSE):
            out = int(inputsY.get_shape()[-1])
            auto_outputY = create_generator_decoder(sR_X2Y, eR_Y2X, out, a, 'autoencoderY_generatorX2Y_decoder_reuse')

    with tf.name_scope("real_discriminatorX2Y"):
        with tf.variable_scope("discriminatorX2Y"):
            predict_realX2Y = create_discriminator(niX, taX, a, 'real_discriminatorX2Y_discriminatorX2Y')
    with tf.name_scope("real_discriminatorY2X"):
        with tf.variable_scope("discriminatorY2X"):
            predict_realY2X = create_discriminator(niY, taY, a, 'real_discriminatorY2X_discriminatorY2X')

    with tf.name_scope("fake_discriminatorX2Y"):
        with tf.variable_scope("discriminatorX2Y", reuse=tf.AUTO_REUSE):
            predict_fakeX2Y = create_discriminator(niX, outputsX2Y, a, 'fake_discriminatorX2Y_discriminatorX2Y')
    with tf.name_scope("fake_discriminatorY2X"):
        with tf.variable_scope("discriminatorY2X", reuse=tf.AUTO_REUSE):
            predict_fakeY2X = create_discriminator(niY, outputsY2X, a, 'fake_discriminatorY2X_discriminatorY2X')

    ######### VISUAL ANALOGIES
    # This is only for visualization (visual analogies), not used in training loss
    with tf.name_scope("image_swapper_X"):
       im_swapped_X,sel_auto_X = create_visual_analogy(sR_X2Y, eR_X2Y,
                                                auto_outputX,inputsX,'Y2X', a)
    with tf.name_scope("image_swapper_Y"):
       im_swapped_Y,sel_auto_Y = create_visual_analogy(sR_Y2X, eR_Y2X,
                                                 auto_outputY,inputsY,'X2Y', a)
######### EXCLUSIVE REPRESENTATION
    # Create generators/discriminators for exclusive representation
    with tf.variable_scope("generator_exclusiveX2Y_decoder"):
        outputs_exclusiveX2Y = create_generator_decoder_exclusive(eR_X2Y, out, a, 'generator_exclusiveX2Y_decoder')
    with tf.name_scope("real_discriminator_exclusiveX2Y"):
        with tf.variable_scope("discriminator_exclusiveX2Y"):
            predict_real_exclusiveX2Y = create_discriminator(niX, taX, a ,'real_discriminator_exclusiveX2Y_discriminator_exclusiveX2Y')
    with tf.name_scope("fake_discriminator_exclusiveX2Y"):
        with tf.variable_scope("discriminator_exclusiveX2Y", reuse=tf.AUTO_REUSE):
            predict_fake_exclusiveX2Y = create_discriminator(niX, outputs_exclusiveX2Y, a ,'fake_discriminator_exclusiveX2Y_discriminator_exclusiveX2Y_reuse')

    with tf.variable_scope("generator_exclusiveY2X_decoder"):
        outputs_exclusiveY2X = create_generator_decoder_exclusive(eR_Y2X, out, a, 'generator_exclusiveY2X_decoder')
    with tf.name_scope("real_discriminator_exclusiveY2X"):
        with tf.variable_scope("discriminator_exclusiveY2X"):
            predict_real_exclusiveY2X = create_discriminator(niY, taY, a, 'real_discriminator_exclusiveY2X_discriminator_exclusiveY2X')
    with tf.name_scope("fake_discriminator_exclusiveY2Y"):
        with tf.variable_scope("discriminator_exclusiveY2X", reuse=tf.AUTO_REUSE):
            predict_fake_exclusiveY2X = create_discriminator(niY, outputs_exclusiveY2X, a, 'fake_discriminator_exclusiveY2Y_discriminator_exclusiveY2X_reuse')

 ######### LOSSES

    __auto_outputX = auto_outputX.outputs
    __auto_outputY = auto_outputY.outputs
    __outputsX2Y = outputsX2Y.outputs
    __outputsY2X = outputsY2X.outputs
    __outputs_exclusiveX2Y = outputs_exclusiveX2Y.outputs
    __outputs_exclusiveY2X = outputs_exclusiveY2X.outputs
    __predict_realX2Y = predict_realX2Y.outputs
    __predict_realY2X = predict_realY2X.outputs
    __predict_fakeX2Y = predict_fakeX2Y.outputs
    __predict_fakeY2X = predict_fakeY2X.outputs
    __predict_real_exclusiveX2Y = predict_real_exclusiveX2Y.outputs
    __predict_real_exclusiveY2X = predict_real_exclusiveY2X.outputs
    __predict_fake_exclusiveX2Y = predict_fake_exclusiveX2Y.outputs
    __predict_fake_exclusiveY2X = predict_fake_exclusiveY2X.outputs

    with tf.name_scope("generatorX2Y_loss"):
        genX2Y_loss_GAN = -tf.reduce_mean(__predict_fakeX2Y)
        genX2Y_loss = genX2Y_loss_GAN * a.gan_weight
    with tf.name_scope("discriminatorX2Y_loss"):
        discrimX2Y_loss = tf.reduce_mean(__predict_fakeX2Y) - tf.reduce_mean(__predict_realX2Y)
        alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
        differences = tf.reshape(__outputsX2Y,[-1,OUTPUT_DIM])-tf.reshape(targetsX,[-1,OUTPUT_DIM])
        interpolates = tf.reshape(targetsX, [-1,OUTPUT_DIM]) + (alpha*differences)
        gradients = tf.gradients(create_discriminator(niX, InputLayer(tf.reshape(interpolates,[-1,IMAGE_SIZE,IMAGE_SIZE,3]), name='albalb1'), a, 'discriminatorX2Y_loss').outputs, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrimX2Y_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generatorY2X_loss"):
        genY2X_loss_GAN = -tf.reduce_mean(__predict_fakeY2X)
        genY2X_loss = genY2X_loss_GAN * a.gan_weight
    with tf.name_scope("discriminatorY2X_loss"):
        discrimY2X_loss = tf.reduce_mean(__predict_fakeY2X) - tf.reduce_mean(__predict_realY2X)
        alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
        differences = tf.reshape(__outputsY2X,[-1,OUTPUT_DIM])-tf.reshape(targetsY,[-1,OUTPUT_DIM])
        interpolates = tf.reshape(targetsY,[-1,OUTPUT_DIM]) + (alpha*differences)
        gradients = tf.gradients(create_discriminator(niY, InputLayer(tf.reshape(interpolates,[-1,IMAGE_SIZE,IMAGE_SIZE,3]), name='albalb2'), a, 'discriminatorY2X_loss').outputs, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrimY2X_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generator_exclusiveX2Y_loss"):
        gen_exclusiveX2Y_loss_GAN = -tf.reduce_mean(__predict_fake_exclusiveX2Y)
        gen_exclusiveX2Y_loss = gen_exclusiveX2Y_loss_GAN * a.gan_exclusive_weight
    with tf.name_scope("discriminator_exclusiveX2Y_loss"):
        discrim_exclusiveX2Y_loss = tf.reduce_mean(__predict_fake_exclusiveX2Y) - tf.reduce_mean(__predict_real_exclusiveX2Y)
        alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
        differences = tf.reshape(__outputs_exclusiveX2Y,[-1,OUTPUT_DIM])-tf.reshape(targetsX,[-1,OUTPUT_DIM])
        interpolates = tf.reshape(targetsX,[-1,OUTPUT_DIM]) + (alpha*differences)
        gradients = tf.gradients(create_discriminator(niX, InputLayer(tf.reshape(interpolates,[-1,IMAGE_SIZE,IMAGE_SIZE,3]), name='albalb3'),a, 'discriminator_exclusiveX2Y_loss').outputs, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrim_exclusiveX2Y_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generator_exclusiveY2X_loss"):
        gen_exclusiveY2X_loss_GAN = -tf.reduce_mean(__predict_fake_exclusiveY2X)
        gen_exclusiveY2X_loss = gen_exclusiveY2X_loss_GAN * a.gan_exclusive_weight
    with tf.name_scope("discriminator_exclusiveY2X_loss"):
        discrim_exclusiveY2X_loss = tf.reduce_mean(__predict_fake_exclusiveY2X) - tf.reduce_mean(__predict_real_exclusiveY2X)
        alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
        differences = tf.reshape(__outputs_exclusiveY2X,[-1,OUTPUT_DIM])-tf.reshape(targetsX,[-1,OUTPUT_DIM])
        interpolates = tf.reshape(targetsX,[-1,OUTPUT_DIM]) + (alpha*differences)
        gradients = tf.gradients(create_discriminator(niX, InputLayer(tf.reshape(interpolates,[-1,IMAGE_SIZE,IMAGE_SIZE,3]), name='albalb4'),a, 'discriminator_exclusiveY2X_loss').outputs, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrim_exclusiveY2X_loss += LAMBDA*gradient_penalty

    with tf.name_scope("autoencoderX_loss"):
        autoencoderX_loss = a.l1_weight*tf.reduce_mean(tf.abs(__auto_outputX-inputsX))
    with tf.name_scope("autoencoderY_loss"):
        autoencoderY_loss = a.l1_weight*tf.reduce_mean(tf.abs(__auto_outputY-inputsY))
    with tf.name_scope("feat_recon_loss"):
        feat_recon_loss = a.l1_weight*tf.reduce_mean(tf.abs(sR_X2Y.outputs-sR_Y2X.outputs))

    with tf.name_scope("code_recon_loss"):
        code_sR_X2Y_recon_loss = tf.reduce_mean(tf.abs(sR_X2Y_recon.outputs-sR_X2Y.outputs))
        code_sR_Y2X_recon_loss = tf.reduce_mean(tf.abs(sR_Y2X_recon.outputs-sR_Y2X.outputs))
        code_eR_X2Y_recon_loss = tf.reduce_mean(tf.abs(eR_X2Y_recon.outputs-z.outputs))
        code_eR_Y2X_recon_loss = tf.reduce_mean(tf.abs(eR_Y2X_recon.outputs-z.outputs))
        code_recon_loss = a.l1_weight*(code_sR_X2Y_recon_loss + code_sR_Y2X_recon_loss + code_eR_X2Y_recon_loss + code_eR_Y2X_recon_loss)

    ######### OPTIMIZERS

    with tf.name_scope("discriminatorX2Y_train"):
        discrimX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorX2Y")]
        discrimX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrimX2Y_grads_and_vars = discrimX2Y_optim.compute_gradients(discrimX2Y_loss, var_list=discrimX2Y_tvars)
        discrimX2Y_train = discrimX2Y_optim.apply_gradients(discrimX2Y_grads_and_vars)

    with tf.name_scope("discriminatorY2X_train"):
        discrimY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorY2X")]
        discrimY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrimY2X_grads_and_vars = discrimY2X_optim.compute_gradients(discrimY2X_loss, var_list=discrimY2X_tvars)
        discrimY2X_train = discrimY2X_optim.apply_gradients(discrimY2X_grads_and_vars)

    with tf.name_scope("generatorX2Y_train"):
        with tf.control_dependencies([discrimX2Y_train]):
            genX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorX2Y")]
            genX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            genX2Y_grads_and_vars = genX2Y_optim.compute_gradients(genX2Y_loss, var_list=genX2Y_tvars)
            genX2Y_train = genX2Y_optim.apply_gradients(genX2Y_grads_and_vars)

    with tf.name_scope("generatorY2X_train"):
        with tf.control_dependencies([discrimY2X_train]):
            genY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorY2X")]
            genY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            genY2X_grads_and_vars = genY2X_optim.compute_gradients(genY2X_loss, var_list=genY2X_tvars)
            genY2X_train = genY2X_optim.apply_gradients(genY2X_grads_and_vars)

    with tf.name_scope("discriminator_exclusiveX2Y_train"):
        discrim_exclusiveX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_exclusiveX2Y")]
        discrim_exclusiveX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_exclusiveX2Y_grads_and_vars = discrim_exclusiveX2Y_optim.compute_gradients(discrim_exclusiveX2Y_loss, var_list=discrim_exclusiveX2Y_tvars)
        discrim_exclusiveX2Y_train = discrim_exclusiveX2Y_optim.apply_gradients(discrim_exclusiveX2Y_grads_and_vars)

    with tf.name_scope("generator_exclusiveX2Y_train"):
        with tf.control_dependencies([discrim_exclusiveX2Y_train]):
            gen_exclusiveX2Y_tvars = [var for var in tf.trainable_variables()
                                      if var.name.startswith("generator_exclusiveX2Y")
                                        or var.name.startswith("generatorX2Y_encoder")]
            gen_exclusiveX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_exclusiveX2Y_grads_and_vars = gen_exclusiveX2Y_optim.compute_gradients(gen_exclusiveX2Y_loss, var_list=gen_exclusiveX2Y_tvars)
            gen_exclusiveX2Y_train = gen_exclusiveX2Y_optim.apply_gradients(gen_exclusiveX2Y_grads_and_vars)

    with tf.name_scope("discriminator_exclusiveY2X_train"):
        discrim_exclusiveY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_exclusiveY2X")]
        discrim_exclusiveY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_exclusiveY2X_grads_and_vars = discrim_exclusiveY2X_optim.compute_gradients(discrim_exclusiveY2X_loss, var_list=discrim_exclusiveY2X_tvars)
        discrim_exclusiveY2X_train = discrim_exclusiveY2X_optim.apply_gradients(discrim_exclusiveY2X_grads_and_vars)

    with tf.name_scope("generator_exclusiveY2X_train"):
        with tf.control_dependencies([discrim_exclusiveY2X_train]):
            gen_exclusiveY2X_tvars = [var for var in tf.trainable_variables()
                                      if var.name.startswith("generator_exclusiveY2X")
                                        or var.name.startswith("generatorY2X_encoder")]
            gen_exclusiveY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_exclusiveY2X_grads_and_vars = gen_exclusiveY2X_optim.compute_gradients(gen_exclusiveY2X_loss, var_list=gen_exclusiveY2X_tvars)
            gen_exclusiveY2X_train = gen_exclusiveY2X_optim.apply_gradients(gen_exclusiveY2X_grads_and_vars)

    with tf.name_scope("autoencoderX_train"):
        autoencoderX_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y_encoder")
                              or var.name.startswith("generatorY2X_encoder")
                              or var.name.startswith("generatorY2X_decoder")]
        autoencoderX_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        autoencoderX_grads_and_vars = autoencoderX_optim.compute_gradients(autoencoderX_loss, var_list=autoencoderX_tvars)
        autoencoderX_train = autoencoderX_optim.apply_gradients(autoencoderX_grads_and_vars)

    with tf.name_scope("autoencoderY_train"):
        autoencoderY_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorY2X_encoder") or
                              var.name.startswith("generatorX2Y_encoder") or
                              var.name.startswith("generatorX2Y_decoder")]
        autoencoderY_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        autoencoderY_grads_and_vars = autoencoderY_optim.compute_gradients(autoencoderY_loss, var_list=autoencoderY_tvars)
        autoencoderY_train = autoencoderY_optim.apply_gradients(autoencoderY_grads_and_vars)


    with tf.name_scope("feat_recon_train"):
        feat_recon_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y_encoder") or
                              var.name.startswith("generatorY2X_encoder")]
        feat_recon_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        feat_recon_grads_and_vars = feat_recon_optim.compute_gradients(feat_recon_loss, var_list=feat_recon_tvars)
        feat_recon_train = feat_recon_optim.apply_gradients(feat_recon_grads_and_vars)

    with tf.name_scope("code_recon_train"):
        code_recon_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y") or
                              var.name.startswith("generatorY2X")]
        code_recon_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        code_recon_grads_and_vars = code_recon_optim.compute_gradients(code_recon_loss, var_list=code_recon_tvars)
        code_recon_train = code_recon_optim.apply_gradients(code_recon_grads_and_vars)




    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrimX2Y_loss, discrimY2X_loss,
                               genX2Y_loss, genY2X_loss,
                               autoencoderX_loss, autoencoderY_loss,
                               feat_recon_loss,code_recon_loss,
                               code_sR_X2Y_recon_loss, code_sR_Y2X_recon_loss,
                               code_eR_X2Y_recon_loss, code_eR_Y2X_recon_loss,
                               discrim_exclusiveX2Y_loss, discrim_exclusiveY2X_loss,
                               gen_exclusiveX2Y_loss, gen_exclusiveY2X_loss])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    return Model(
        predict_realX2Y=predict_realX2Y.outputs,
        predict_realY2X=predict_realY2X.outputs,
        predict_fakeX2Y=predict_fakeX2Y.outputs,
        predict_fakeY2X=predict_fakeY2X.outputs,
        im_swapped_X=im_swapped_X.outputs,
        im_swapped_Y=im_swapped_Y.outputs,
        sel_auto_X=sel_auto_X,
        sel_auto_Y=sel_auto_Y,
        sR_X2Y=sR_X2Y.outputs,
        sR_Y2X=sR_Y2X.outputs,
        eR_X2Y=eR_X2Y.outputs,
        eR_Y2X=eR_Y2X.outputs,
        discrimX2Y_loss=ema.average(discrimX2Y_loss),
        discrimY2X_loss=ema.average(discrimY2X_loss),
        genX2Y_loss=ema.average(genX2Y_loss),
        genY2X_loss=ema.average(genY2X_loss),
        discrim_exclusiveX2Y_loss=ema.average(discrim_exclusiveX2Y_loss),
        discrim_exclusiveY2X_loss=ema.average(discrim_exclusiveY2X_loss),
        gen_exclusiveX2Y_loss=ema.average(gen_exclusiveX2Y_loss),
        gen_exclusiveY2X_loss=ema.average(gen_exclusiveY2X_loss),
        outputsX2Y=outputsX2Y.outputs,
        outputsY2X=outputsY2X.outputs,
        outputsX2Yp=outputsX2Yp.outputs,
        outputsY2Xp=outputsY2Xp.outputs,
        outputs_exclusiveX2Y=outputs_exclusiveX2Y.outputs,
        outputs_exclusiveY2X=outputs_exclusiveY2X.outputs,
        auto_outputX = auto_outputX.outputs,
        autoencoderX_loss=ema.average(autoencoderX_loss),
        auto_outputY = auto_outputY.outputs,
        autoencoderY_loss=ema.average(autoencoderY_loss),
        feat_recon_loss=ema.average(feat_recon_loss),
        code_recon_loss=ema.average(code_recon_loss),
        code_sR_X2Y_recon_loss=ema.average(code_sR_X2Y_recon_loss),
        code_sR_Y2X_recon_loss=ema.average(code_sR_Y2X_recon_loss),
        code_eR_X2Y_recon_loss=ema.average(code_eR_X2Y_recon_loss),
        code_eR_Y2X_recon_loss=ema.average(code_eR_Y2X_recon_loss),
        train=tf.group(update_losses, incr_global_step, genX2Y_train,
                       genY2X_train, autoencoderX_train, autoencoderY_train,code_recon_train,
                       gen_exclusiveX2Y_train,gen_exclusiveY2X_train,feat_recon_train),
    )
def create_visual_analogy(__sR, __eR, auto_output, inputs, which_direction, a):
    sR=__sR.outputs
    eR=__eR.outputs
    swapScoreBKG = 0
    sR_Swap = []   
    eR_Swap = []
    sel_auto = []

    for i in range(0,a.batch_size):
        s_curr = tf.reshape(sR[i,:],[sR.shape[1],sR.shape[2],sR.shape[3]])
        # Take a random image from the batch, make sure it is different from current
        bkg_ims_idx = random.randint(0,a.batch_size-1)
        while bkg_ims_idx == i:
            bkg_ims_idx = random.randint(0,a.batch_size-1)
        ex_rnd = tf.reshape(eR[bkg_ims_idx,:],[eR.shape[1]])
        sR_Swap.append(s_curr)
        eR_Swap.append(ex_rnd)
        # Store also selected reference image for visualization
        sel_auto.append(inputs[bkg_ims_idx,:])

    with tf.variable_scope("generator" + which_direction + "_decoder", reuse=tf.AUTO_REUSE):
        out_channels = int(auto_output.outputs.get_shape()[-1])
        im_swapped = create_generator_decoder(InputLayer(tf.stack(sR_Swap), name='djdj1'),
                                                    InputLayer(tf.stack(eR_Swap), name='djdj2'),
                                                    out_channels, a, 'djdj3')
                                                        
    return im_swapped, tf.stack(sel_auto)
