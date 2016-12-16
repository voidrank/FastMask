import sys
import os
import argparse
import time
import cjson
from math import ceil
import numpy as np
from IPython import embed
import setproctitle 
import cv2

sys.path.append(os.path.abspath("caffe-fm/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe

from alchemy.utils.image import resize_blob, visualize_masks
from alchemy.utils.timer import Timer
from alchemy.utils.mask import encode, crop
from alchemy.utils.load_config import load_config

import config

'''
    python demo.py gpu_id model_prototxt [--debug=False] [--init_weights=*.caffemodel] [--display=mask] [--start_from=0] [--start_scale=0]
'''




def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model_prototxt', type=str)
    parser.add_argument('--debug', dest='debug', type=bool, default=False)
    parser.add_argument('--init_weights', dest='init_weights', type=str,
                        default=None)
    parser.add_argument('--display', dest='display', type=str,
                        default='mask')
    parser.add_argument('--start_from', dest='start_from', type=int,
                        default=0)
    parser.add_argument('--start_scale', dest='start_scale', type=int,
                        default=0)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    setproctitle.setproctitle(args.model_prototxt)

    net = caffe.Net(
            'models/' + args.model_prototxt + ".test.prototxt",
            'params/' + args.init_weights,
            caffe.TEST)

    if os.path.exists("configs/%s.json" % args.model_prototxt):
        load_config("configs/%s.json" % args.model_prototxt)
    else:
        print "Specified config does not exists, use the default config..."

    config.ANNOTATION_TYPE = "val2014"
    config.IMAGE_SET = "val2014"
    from spiders.coco_ssm_spider import COCOSSMDemoSpider
    spider = COCOSSMDemoSpider()
    ds = spider.dataset

    i = 0
    while i < args.start_from:
        spider.next_idx()
        i += 1

    timer = Timer()

    for i in range(i, len(ds)):
        if len(ds[i].imgToAnns) == 0:
            continue
        else:
            image_id = ds[i].imgToAnns[0]['image_id']
        spider.fetch()
        img = spider.img_blob
        oh, ow = img.shape[2:]
        net.blobs['data'].reshape(*img.shape)
        net.blobs['data'].data[...] = img
        timer.tic()
        net.forward()
        print timer.tac()

        stride = 16
        h, w = (oh/stride) + (oh%stride > 0), (ow/stride) + (ow%stride > 0)
        ratio = 16
        ratios = spider.RFs
        try:
            ratios = config.TEST_RFs
        except Exception:
            pass
        scales = []
        for rf in ratios:
            scales.append(((oh/rf)+(oh%rf>0), (ow/rf)+(ow%rf>0)))

        order = net.blobs['top_k'].data.flatten()
        embed()
        for _ in range(len(net.blobs['top_k'].data[:,0,0,0])):
            bid = int(order[_])
            print net.blobs['objn'].data[bid]
            print net.blobs['objn'].data[bid].argmax()

            scale_idx = 0
            h, w = scales[scale_idx]
            ceiling = (h + 1) * (w + 1)
            scale_idx = 0
            while bid >= ceiling:
                scale_idx += 1
                h, w = scales[scale_idx]
                bid -= ceiling
                ceiling = (h + 1) * (w + 1)

            stride = ratios[scale_idx]
            if stride < args.start_scale:
                continue

            print 'stride: ', stride

            x = bid / (w + 1)
            y = bid % (w + 1)


            SLIDING_WINDOW_SIZE = config.SLIDING_WINDOW_SIZE
            xb, xe = int(round((x - SLIDING_WINDOW_SIZE/2) * stride)), int(round((x + SLIDING_WINDOW_SIZE/2) * stride))
            yb, ye = int(round((y - SLIDING_WINDOW_SIZE/2) * stride)), int(round((y + SLIDING_WINDOW_SIZE/2) * stride))
            size = xe - xb, ye - yb

            if args.display == 'mask':
                masks = net.blobs['masks'].data[_]
                masks[masks > 0.2] = 1
                masks[masks <= 0.2] = 0
            else:
                masks = net.blobs['atts'].data[_]
            masks = resize_blob(masks, size, method=cv2.INTER_LANCZOS4)
            masks = crop(masks, (xb, xe, yb, ye), (oh, ow))

            RGB_MEAN = config.RGB_MEAN
            img = net.blobs['data'].data[0, :, max(xb, 0): min(oh, xe), max(yb, 0): min(ow, ye)].copy()
            img = img.transpose((1, 2, 0))
            img += RGB_MEAN

            visualize_masks(img, masks)
