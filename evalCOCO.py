import sys
import os
import argparse
import time
import cjson
import threading
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
from IPython import embed
import config
from config import *

from pycocotools.cocoeval import COCOeval

from alchemy.utils.image import resize_blob, visualize_masks
from alchemy.utils.timer import Timer
from alchemy.utils.mask import encode, crop, iou
from alchemy.utils.progress_bar import printProgress


'''
python evalCOCO.py model [--debug=False] [--useDet=False] [--useSegm=True] \
        [--max_proposal=100] [--dataset=val2014] [--debug=False] [--end=5000] \
        [--nms_threshold=0.7] [--objn_threshold=0]
'''

import numpy as np

import setproctitle 

import cv2


def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('model', type=str)
    parser.add_argument('--useCats', dest='useCats', type=str, default='False')
    parser.add_argument('--useSegm', dest='useSegm', type=str, default='True')
    parser.add_argument('--end', dest='end', type=int, default=5000)
    parser.add_argument('--debug', dest='debug', type=str, default='False')
    parser.add_argument('--nms_threshold', dest='nms_threshold', type=float, default=.7)
    parser.add_argument('--dataset', dest='dataset', type=str, default='val2014')
    parser.add_argument('--objn_threshold', dest='objn_threshold', type=float, default=0)
    parser.add_argument('--max_proposal', dest='max_proposal', type=int, default=100)

    args = parser.parse_args()
    args.useSegm = args.useSegm == 'True'
    args.useCats = args.useCats == 'True'
    args.debug = args.debug == 'True'
    return args

if __name__ == '__main__':
    args = parse_args()

    max_dets = [1, 10, 100, 1000]
    i = 1

    with open('results/%s.json' % args.model, 'rb') as f:
        input_results = cjson.decode(f.read())
        results = []
        _ = 0
        while _ < len(input_results):
            sub_results = []
            start = _
            while _ < len(input_results) and input_results[start]['image_id'] == input_results[_]['image_id']:
                sub_results.append(input_results[_])
                _ += 1

            printProgress(_, len(input_results), 'Results preprocess: ', suffix = 'Complete', barLength = 50)

            sub_results.sort(key=lambda item: item['objn'], reverse=True)
            # nms
            keep = np.ones(len(sub_results)).astype(np.bool)
            if args.nms_threshold < 1:
                for i in range(len(sub_results)):
                    if keep[i]:
                        for j in range(i+1, len(sub_results)):
                            if keep[j] and iou(sub_results[i]['segmentation'], sub_results[j]['segmentation'], [False]) > args.nms_threshold:
                                keep[j] = False

            # objn 
            if args.objn_threshold > 0:
                for i in range(len(sub_results)):
                    if sub_results[i]['objn'] < args.objn_threshold:
                        keep[i] = False

            for i in reversed(np.where(keep==False)[0]):
                del sub_results[i]
            sub_results = sub_results[:args.max_proposal]

            for result in sub_results:
                results.append(result)

    with open("results/%s_temp.json" % args.model, 'wb') as f:
        f.write(cjson.encode(results))


    config.ANNOTATION_TYPE = args.dataset
    config.IMAGE_SET = args.dataset
    from spiders.coco_ssm_spider import COCOSSMDemoSpider
    spider = COCOSSMDemoSpider()
    ds = spider.dataset

    cocoGt = ds
    cocoDt = cocoGt.loadRes("results/%s_temp.json" % args.model)
    cocoEval = COCOeval(cocoGt, cocoDt)

    if args.debug:
        embed()

    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())[:args.end]
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = args.useSegm
    cocoEval.params.useCats = args.useCats
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
