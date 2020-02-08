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

from model import create_model

import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
# Added features mode to extract features for retrieval
parser.add_argument("--mode", required=True, choices=["train", "test", "features"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=30, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# Cross-domain-disen new arugments
parser.add_argument("--gan_exclusive_weight", type=float, default=0.1, help="weight on GAN term for exclusive generator gradient")
parser.add_argument("--noise", type=float, default=0.1, help="Stddev for noise input into representation")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputsX, inputsY, count, steps_per_epoch")


def preprocess(image):
    with tf.name_scope("preprocess"):
       return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])



def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def mlp(input, out_dim, dtype=tf.float32, bias=True):
        _, n = input.get_shape()
        w = tf.get_variable('w', [n, out_dim], dtype, tf.random_normal_initializer(0.0, 0.02))
        out = tf.matmul(input, w)
        if bias:
            b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
            out = out + b
        return out


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])

    # No longer in terms of input/target, but bidirectionally on domains X and Y
    inputsX, inputsY = [a_images, b_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("inputX_images"):
        inputX_images = transform(inputsX)

    with tf.name_scope("inputY_images"):
        inputY_images = transform(inputsY)

    paths_batch, inputsX_batch, inputsY_batch = tf.train.batch([paths,inputX_images,inputY_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputsX=inputsX_batch,
        inputsY=inputsY_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        all_kinds = ["inputsX", "outputsX2Y", "outputsX2Yp",
                         "auto_outputsX","im_swapped_X", "sel_auto_X","inputsY",
                         "outputsY2X", "outputsY2Xp","auto_outputsY" ,"im_swapped_Y", "sel_auto_Y"]

        for kind in all_kinds:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def save_features(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "features")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        filename = name + ".mat"
        out_path = os.path.join(image_dir, filename)
        sio.savemat(out_path,{'inX':fetches["inputsX"][i],
                             'inY':fetches["inputsY"][i],
                             'sR_X2Y':fetches["sR_X2Y"][i],
                             'sR_Y2X':fetches["sR_Y2X"][i],
                             'eR_X2Y':fetches["eR_X2Y"][i],
                             'eR_Y2X':fetches["eR_Y2X"][i]})


    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>inX</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>swap</th><th>randomimage</th><th>inY</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>swap</th><th>rnd</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
        all_kinds = ["inputsX", "outputsX2Y", "outputsX2Yp",
                     "auto_outputsX","im_swapped_X", "sel_auto_X","inputsY",
                     "outputsY2X", "outputsY2Xp","auto_outputsY" ,"im_swapped_Y", "sel_auto_Y"]

        for kind in all_kinds:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "features":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputsX, examples.inputsY, a)

    print(model)

    # undo colorization splitting on images that we use for display/output
    inputsX = deprocess(examples.inputsX)
    inputsY = deprocess(examples.inputsY)
    outputsX2Y = deprocess(model.outputsX2Y)
    outputsY2X = deprocess(model.outputsY2X)
    outputsX2Yp = deprocess(model.outputsX2Yp)
    outputsY2Xp = deprocess(model.outputsY2Xp)
    outputs_exclusiveX2Y = deprocess(model.outputs_exclusiveX2Y)
    outputs_exclusiveY2X = deprocess(model.outputs_exclusiveY2X)
    auto_outputX = deprocess(model.auto_outputX)
    auto_outputY = deprocess(model.auto_outputY)
    im_swapped_X = deprocess(model.im_swapped_X)
    im_swapped_Y = deprocess(model.im_swapped_Y)
    sel_auto_X = deprocess(model.sel_auto_X)
    sel_auto_Y = deprocess(model.sel_auto_Y)
    sR_X2Y = model.sR_X2Y
    sR_Y2X = model.sR_Y2X
    eR_X2Y = model.eR_X2Y
    eR_Y2X = model.eR_Y2X

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputsX"):
        converted_inputsX = convert(inputsX)

    with tf.name_scope("convert_inputsY"):
        converted_inputsY = convert(inputsY)

    with tf.name_scope("convert_outputsX2Y"):
        converted_outputsX2Y = convert(outputsX2Y)

    with tf.name_scope("convert_outputsY2X"):
        converted_outputsY2X = convert(outputsY2X)

    with tf.name_scope("convert_outputsX2Yp"):
        converted_outputsX2Yp = convert(outputsX2Yp)

    with tf.name_scope("convert_outputsY2Xp"):
        converted_outputsY2Xp = convert(outputsY2Xp)

    with tf.name_scope("convert_outputs_exclusiveX2Y"):
        converted_outputs_exclusiveX2Y = convert(outputs_exclusiveX2Y)

    with tf.name_scope("convert_outputs_exclusiveY2X"):
        converted_outputs_exclusiveY2X = convert(outputs_exclusiveY2X)

    with tf.name_scope("convert_auto_outputsX"):
        converted_auto_outputX = convert(auto_outputX)

    with tf.name_scope("convert_auto_outputsY"):
        converted_auto_outputY = convert(auto_outputY)

    with tf.name_scope("convert_im_swapped_Y"):
        converted_im_swapped_Y = convert(im_swapped_Y)

    with tf.name_scope("convert_sel_auto_Y"):
        converted_sel_auto_Y= convert(sel_auto_Y)

    with tf.name_scope("convert_im_swapped_X"):
        converted_im_swapped_X = convert(im_swapped_X)

    with tf.name_scope("convert_sel_auto_X"):
        converted_sel_auto_X= convert(sel_auto_X)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputsX": tf.map_fn(tf.image.encode_png, converted_inputsX, dtype=tf.string, name="inputX_pngs"),
            "inputsY": tf.map_fn(tf.image.encode_png, converted_inputsY, dtype=tf.string, name="inputY_pngs"),
            "outputsX2Y": tf.map_fn(tf.image.encode_png, converted_outputsX2Y, dtype=tf.string, name="outputX2Y_pngs"),
            "outputsY2X": tf.map_fn(tf.image.encode_png, converted_outputsY2X, dtype=tf.string, name="outputY2X_pngs"),
            "outputsX2Yp": tf.map_fn(tf.image.encode_png, converted_outputsX2Yp, dtype=tf.string, name="outputX2Yp_pngs"),
            "outputsY2Xp": tf.map_fn(tf.image.encode_png, converted_outputsY2Xp, dtype=tf.string, name="outputY2Xp_pngs"),
            "outputs_exclusiveX2Y": tf.map_fn(tf.image.encode_png, converted_outputs_exclusiveX2Y, dtype=tf.string, name="output_exclusiveX2Y_pngs"),
            "outputs_exclusiveY2X": tf.map_fn(tf.image.encode_png, converted_outputs_exclusiveY2X, dtype=tf.string, name="output_exclusiveY2X_pngs"),
            "auto_outputsX": tf.map_fn(tf.image.encode_png, converted_auto_outputX, dtype=tf.string, name="auto_outputX_pngs"),
            "auto_outputsY": tf.map_fn(tf.image.encode_png, converted_auto_outputY, dtype=tf.string, name="auto_outputY_pngs"),
            "im_swapped_Y": tf.map_fn(tf.image.encode_png, converted_im_swapped_Y, dtype=tf.string, name="im_swapped_Y_pngs"),
            "sel_auto_Y": tf.map_fn(tf.image.encode_png, converted_sel_auto_Y, dtype=tf.string, name="sel_auto_Y_pngs"),
            "im_swapped_X": tf.map_fn(tf.image.encode_png, converted_im_swapped_X, dtype=tf.string, name="im_swapped_X_pngs"),
            "sel_auto_X": tf.map_fn(tf.image.encode_png, converted_sel_auto_X, dtype=tf.string, name="sel_auto_X_pngs"),
        }
    with tf.name_scope("extract_features"):
        features_fetches = {
            "paths": examples.paths,
            "inputsX": converted_inputsX,
            "sR_X2Y": sR_X2Y,
            "eR_X2Y": eR_X2Y,
            "inputsY": converted_inputsY,
            "sR_Y2X": sR_Y2X,
            "eR_Y2X": eR_Y2X,
        }

    # summaries
    with tf.name_scope("X1_input_summary"):
        tf.summary.image("inputsX", converted_inputsX,max_outputs=3)

    with tf.name_scope("Y1_input_summary"):
        tf.summary.image("inputsY", converted_inputsY,max_outputs=3)

    with tf.name_scope("X2Y_output_summary"):
        tf.summary.image("outputsX2Y", converted_outputsX2Y,max_outputs=3)

    with tf.name_scope("Y2X_outpu2_summary"):
        tf.summary.image("outputsY2X", converted_outputsY2X,max_outputs=3)

    with tf.name_scope("X_autoencoder_summary"):
        tf.summary.image("auto_outputX", converted_auto_outputX,max_outputs=3)

    with tf.name_scope("Y_autoencoder_summary"):
        tf.summary.image("auto_outputY", converted_auto_outputY,max_outputs=3)

    with tf.name_scope("swapped_1Y_summary"):
        tf.summary.image("im_swapped_Y", converted_im_swapped_Y,max_outputs=3)
        tf.summary.image("sel_auto_Y", converted_sel_auto_Y,max_outputs=3)

    with tf.name_scope("swapped_2X_summary"):
        tf.summary.image("im_swapped_X", converted_im_swapped_X,max_outputs=3)
        tf.summary.image("sel_auto_X", converted_sel_auto_X,max_outputs=3)

    with tf.name_scope("otherNoise_output_summary"):
        tf.summary.image("outputsX2Yp", converted_outputsX2Yp,max_outputs=3)
        tf.summary.image("outputsY2Xp", converted_outputsY2Xp,max_outputs=3)

    with tf.name_scope("zzexclusive_X2Y_summary"):
        tf.summary.image("outputsX2Y", converted_outputs_exclusiveX2Y,max_outputs=3)

    with tf.name_scope("zzexclusive_Y2X_summary"):
        tf.summary.image("outputsY2X", converted_outputs_exclusiveY2X,max_outputs=3)

    tf.summary.scalar("discriminatorX2Y_loss", model.discrimX2Y_loss)
    tf.summary.scalar("discriminatorY2X_loss", model.discrimY2X_loss)
    tf.summary.scalar("generatorX2Y_loss", model.genX2Y_loss)
    tf.summary.scalar("generatorY2X_loss", model.genY2X_loss)
    tf.summary.scalar("generator_exclusiveX2Y_loss", model.gen_exclusiveX2Y_loss)
    tf.summary.scalar("discriminator_exclusiveX2Y_loss", model.discrim_exclusiveX2Y_loss)
    tf.summary.scalar("generator_exclusiveY2X_loss", model.gen_exclusiveY2X_loss)
    tf.summary.scalar("discriminator_exclusiveY2X_loss", model.discrim_exclusiveY2X_loss)
    tf.summary.scalar("autoencoderX_loss", model.autoencoderX_loss)
    tf.summary.scalar("autoencoderY_loss", model.autoencoderY_loss)
    tf.summary.scalar("feat_recon_loss", model.feat_recon_loss)
    tf.summary.scalar("code_sR_X2Y_recon_loss", model.code_sR_X2Y_recon_loss)
    tf.summary.scalar("code_sR_Y2X_recon_loss", model.code_sR_Y2X_recon_loss)
    tf.summary.scalar("code_eR_X2Y_recon_loss", model.code_eR_X2Y_recon_loss)
    tf.summary.scalar("code_eR_Y2X_recon_loss", model.code_eR_Y2X_recon_loss)
    tf.summary.scalar("code_recon_loss", model.code_recon_loss)

    #for var in tf.trainable_variables():
        #tf.summary.histogram(var.op.name + "/values", var)

    #for grad, var in model.discrimX2Y_grads_and_vars + model.genX2Y_grads_and_vars:
        #tf.summary.histogram(var.op.name + "/gradientsX2Y", grad)

    #for grad, var in model.discrimY2X_grads_and_vars + model.genY2X_grads_and_vars:
        #tf.summary.histogram(var.op.name + "/gradientsY2X", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        elif a.mode == "features":
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(features_fetches)
                save_features(results)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()


                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrimX2Y_loss"] = model.discrimX2Y_loss
                    fetches["discrimY2X_loss"] = model.discrimY2X_loss
                    fetches["genX2Y_loss"] = model.genX2Y_loss
                    fetches["genY2X_loss"] = model.genY2X_loss
                    fetches["autoencoderX_loss"] = model.autoencoderX_loss
                    fetches["autoencoderY_loss"] = model.autoencoderY_loss
                    fetches["code_recon_loss"] = model.code_recon_loss
                    fetches["feat_recon_loss"] = model.feat_recon_loss

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrimX2Y_loss", results["discrimX2Y_loss"])
                    print("discrimY2X_loss", results["discrimY2X_loss"])
                    print("genX2Y_loss", results["genX2Y_loss"])
                    print("genY2X_loss", results["genY2X_loss"])
                    print("autoencoderX_loss", results["autoencoderX_loss"])
                    print("autoencoderY_loss", results["autoencoderY_loss"])
                    print("code_recon_loss", results["code_recon_loss"])
                    print("feat_recon_loss", results["feat_recon_loss"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
