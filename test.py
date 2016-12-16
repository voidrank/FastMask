import sys
import os
import argparse
import time
import cjson
import threading
sys.path.append(os.path.abspath("caffe-fm/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe
from IPython import embed
import numpy as np
import setproctitle 
import cv2


from pycocotools.cocoeval import COCOeval

from alchemy.utils.image import resize_blob, visualize_masks
from alchemy.utils.timer import Timer
from alchemy.utils.mask import encode, crop, iou
from alchemy.utils.load_config import load_config
from alchemy.utils.progress_bar import printProgress


import config
import utils
from utils import gen_masks
from config import *



'''
python test.py gpu_id model [--debug=False] [--init_weights=*.caffemodel] [--useDet=False] \
        [--test_num=5000] [--dataset=val2014] [--debug=False] [--end=5000] \
'''



def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('--useCats', dest='useCats', type=str, default='False')
    parser.add_argument('--debug', dest='debug', type=str, default='False')
    parser.add_argument('--init_weights', dest='init_weights', type=str,
                        default=None)
    parser.add_argument('--dataset', dest='dataset', type=str, 
                        default='val2014')
    parser.add_argument('--end', dest='end', type=int, default=5000)


    args = parser.parse_args()
    args.useCats = args.useCats == 'True'
    args.debug = args.debug == 'True'
    return args



if __name__ == '__main__':
    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    setproctitle.setproctitle(args.model)

    net = caffe.Net(
            'models/' + args.model + ".test.prototxt",
            'params/' + args.init_weights,
            caffe.TEST)

    # surgeries
    interp_layers = [layer for layer in net.params.keys() if 'up' in layer]
    utils.interp(net, interp_layers)

    if os.path.exists("configs/%s.json" % args.model):
        load_config("configs/%s.json" % args.model)
    else:
        print "Specified config does not exists, use the default config..."

    time.sleep(2)

    timer = Timer()

    config.ANNOTATION_TYPE = args.dataset
    config.IMAGE_SET = "val2014"
    from spiders.coco_ssm_spider import COCOSSMDemoSpider
    spider = COCOSSMDemoSpider()
    spider.dataset.sort(key=lambda item: int(item.image_path[-10:-4]))
    ds = spider.dataset[:args.end]

    timer.tic()
    results = []
    for i in range(len(ds)):

        spider.fetch()
        img = spider.img_blob
        image_id = int(ds[i].image_path[-10:-4])

        # gen mask
        ret = gen_masks(net, img, config, 
                dest_shape=(spider.origin_height, spider.origin_width), 
                useCats=args.useCats, vis=args.debug)

        if args.useCats:
            ret_masks, ret_scores, ret_cats = ret
        else:
            ret_masks, ret_scores = ret

        printProgress(i, len(ds), prefix='Progress: ', suffix='Complete', barLength=50)
        for _ in range(len(ret_masks)):
            cat = 1
            if args.useCats:
                cat = spider.dataset.getCatIds()[int(ret_cats[_].argmax())]
                score = float(ret_cats[_].max())
            else:
                score = float(ret_scores[_])
            objn = float(ret_scores[_])
            results.append({
                'image_id': image_id,
                'category_id': cat,
                'segmentation': encode(ret_masks[_]),
                'score': score,
                'objn': objn
                })


    with open('results/%s.json' % args.model, "wb") as f:
        f.write(cjson.encode(results))

