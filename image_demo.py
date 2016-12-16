import sys
import os
import argparse
import time
import cjson
from math import ceil
sys.path.append(os.path.abspath("caffe-fm/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe
from IPython import embed
import config

import numpy as np
import setproctitle
import cv2

from alchemy.utils.image import resize_blob, visualize_masks, load_image
from alchemy.utils.timer import Timer
from alchemy.utils.mask import encode, decode, crop, iou
from alchemy.utils.load_config import load_config

from utils import gen_masks


'''
    python image_demo.py gpu_id model input_image
'''


COLORS = [0xE6E2AF, 0xA7A37E, 0xDC3522, 0x046380, 
        0x468966, 0xB64926, 0x8E2800, 0xFFE11A,
        0xFF6138, 0x193441, 0xFF9800, 0x7D9100,
        0x1F8A70, 0x7D8A2E, 0x2E0927, 0xACCFCC,
        0x644D52, 0xA49A87, 0x04BFBF, 0xCDE855,
        0xF2836B, 0x88A825, 0xFF358B, 0x01B0F0,
        0xAEEE00, 0x334D5C, 0x45B29D, 0xEFC94C,
        0xE27A3F, 0xDF5A49]


def parse_args():
    parser = argparse.ArgumentParser('process image')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('input_image', type=str)
    parser.add_argument('--init_weights', type=str,
                        default='', dest='init_weights')
    parser.add_argument('--threshold', type=float,
                        default=0.90, dest='threshold')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # caffe setup
    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))

    net = caffe.Net(
            'models/' + args.model + '.test.prototxt',
            'params/' + args.init_weights,
            caffe.TEST)

    # load config
    if os.path.exists("configs/%s.json" % args.model):
        load_config("configs/%s.json" % args.model)
    else:
        print "Specified config does not exists, use the default config..."

    image = load_image(args.input_image)
    oh, ow = image.shape[:2]
    im_scale = config.TEST_SCALE * 1.0 / max(oh, ow)
    input_blob = image - config.RGB_MEAN
    input_blob = input_blob.transpose((2, 0, 1))
    ih, iw = int(oh * im_scale), int(ow * im_scale)
    ih, iw = ih - ih % 4, iw - iw % 4
    input_blob = resize_blob(input_blob, dest_shape=(ih, iw))
    input_blob = input_blob[np.newaxis, ...]

    ret_masks, ret_scores = gen_masks(net, input_blob, config, dest_shape=(oh, ow))


    # nms
    encoded_masks = encode(ret_masks)
    reserved = np.ones((len(ret_masks)))
    for i in range(len(reserved)):
        if ret_scores[i] < args.threshold:
            reserved[i] = 0
            continue
        if reserved[i]:
            for j in range(i + 1, len(reserved)):
                if reserved[j] and iou(encoded_masks[i], encoded_masks[j], [False]) > 0.5:
                    reserved[j] = 0


    for _ in range(len(ret_masks)):
        if ret_scores[_] > args.threshold and reserved[_]:
            mask = ret_masks[_].copy()
            mask[mask == 1] = 0.3
            mask[mask == 0] = 1
            color = COLORS[_ % len(COLORS)]
            for k in range(3):
                image[:,:,k] = image[:,:,k] * mask
            mask[mask == 1] = 0
            mask[mask > 0] = 0.7
            for k in range(3):
                image[:,:,k] += mask * (color & 0xff)
                color >>= 8;

    image = image.astype(np.uint8)
    cv2.imshow('image', image)
    cv2.waitKey(100000)


