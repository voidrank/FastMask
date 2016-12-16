import numpy as np
import os
import cv2
from math import ceil, floor
from config import *

from alchemy.spiders.dataset_spider import DatasetSpider
from alchemy.datasets.coco import COCO_DS

from pycocotools.mask import frPyObjects

from alchemy.utils.image import (load_image, image_to_data, sub_mean,
                        resize_blob, draw_attention, visualize_bbs,
                        visualize_masks)
from alchemy.utils.mask import (decode, area, toBbox, pts_in_bbs, bbs_in_bbs,
                        encode)

from base_coco_ssm_spider import *



class COCOSSMSpider(BaseCOCOSSMSpider):

    attr = ['image', 'gt_objns', 'gt_cats', 'gt_masks', 'gt_atts', 'objn_filter', 'cat_filter', 'mask_filter'] \
            if config.USE_CATS else \
           ['image', 'gt_objns', 'gt_masks', 'gt_atts', 'objn_filter', 'mask_filter']

    random_idx = True
    flipped = True
    tiny_zoom = range(70)

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
            # 0 for '__background__'
            self.__class__.cats_to_labels = dict([(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMSpider, self).__init__(*args, **kwargs)
        self.RFs = RFs
        self.SCALE = SCALE


    def fetch(self):

        while True:
            SCALE = self.SCALE
            if self.__class__.flipped:
                self.flipped = bool(np.random.choice(2, 1))
            if self.tiny_zoom is not None:
                tiny_zoom = np.random.choice(self.__class__.tiny_zoom, 1)
            else:
                tiny_zoom = 0
            self.max_edge = tiny_zoom + SCALE
            idx = self.get_idx()
            item = self.dataset[idx]
            self.image_path = item.image_path
            self.anns = item.imgToAnns

            batch = {}

            batch.update(self.fetch_image())
            self.fetch_masks()
            self.cal_centers_of_masks_and_bbs()

            self.zoom_negtive_samples = None
            self.shift_negtive_samples = None
            try:
                batch.update(self.fetch_label())
            except NoLabelException:
                continue

            batch = self.filter_sample(batch)
            
            return batch


    def fetch_label(self):

        gt_objns = []
        mask_filter = []
        gt_masks = []
        gt_atts = []
        if config.USE_CATS:
            gt_cats = []

        for rf in self.RFs:
            h, w, ratio = (self.height/rf) + (self.height%rf>0), (self.width/rf) + (self.width%rf>0), rf
            self.gen_single_scale_label(h, w, ratio)
            gt_objns.append(self.gt_objns.copy())
            if config.USE_CATS:
                gt_cats.append(self.gt_cats.copy())
            mask_filter.append(self.mask_filter.copy())
            gt_masks.append(self.gt_masks.copy())
            gt_atts.append(self.gt_atts.copy())

        gt_objns = np.concatenate(gt_objns)
        if config.USE_CATS:
            gt_cats = np.concatenate(gt_cats)
        mask_filter = np.concatenate(mask_filter)
        gt_masks = np.concatenate(gt_masks)
        gt_atts = np.concatenate(gt_atts)
        gt_masks = gt_masks[mask_filter == 1]
        gt_atts = gt_atts[mask_filter == 1]

        # vis
        # self.gen(SCALES[0], True)
        # print self.bbs_hw[match == 0]

        if gt_objns.max() < 1:
            print 'ignore'
            raise NoLabelException("No label")

        ret = {
                'gt_objns': gt_objns,
                'gt_masks': gt_masks,
                'gt_atts':  gt_atts,
                'mask_filter': mask_filter
                }

        if config.USE_CATS:
            ret['gt_cats'] = gt_cats

        return ret


    def gen_single_scale_label(self, h, w, ratio):
        self.feat_h = h + 1
        self.feat_w = w + 1
        self.ratio = ratio
        self.find_matched_masks()
        self.assign_gt()


    def filter_sample(self, batch):
        self.negtive_samples = (self.zoom_negtive_samples == 1) | (self.shift_negtive_samples == 1)

        positive_num = np.count_nonzero(batch['gt_objns'])
        positive_samples = batch['gt_objns']
        positive_sample_ids = np.where(batch['gt_objns'])[0]
        if positive_num * 2 > OBJN_BATCH_SIZE:
            positive_sample_ids = np.random.choice(positive_sample_ids, OBJN_BATCH_SIZE/2, replace=False)
            positive_samples[...] = 0
            positive_samples[positive_sample_ids] = 1
            positive_num = OBJN_BATCH_SIZE/2

        negtive_samples = self.negtive_samples
        negtive_sample_ids = np.where(negtive_samples)[0]
        negtive_num = len(negtive_sample_ids)
        if negtive_num * 2 > OBJN_BATCH_SIZE:
            negtive_sample_ids = np.random.choice(negtive_sample_ids, OBJN_BATCH_SIZE/2, replace=False)
            negtive_samples[...] = 0
            negtive_samples[negtive_sample_ids] = 1
            negtive_num = OBJN_BATCH_SIZE/2

        '''
        if positive_num + negtive_num < OBJN_BATCH_SIZE:
            rest = np.where((positive_samples == 0) & (negtive_samples == 0))[0]
            rest_samples_ids = np.random.choice(rest, OBJN_BATCH_SIZE - positive_num - negtive_num, replace=False)
            negtive_samples[rest_samples_ids] = 1
            negtive_num = OBJN_BATCH_SIZE - positive_num
        '''

        objn_filter = np.zeros(len(positive_samples))
        objn_filter[(positive_samples > 0)| (negtive_samples > 0)] = 1
        self.positive_samples = positive_samples
        self.negtive_samples = negtive_samples
        batch['objn_filter'] = objn_filter
        batch['gt_objns'] = batch['gt_objns'][objn_filter > 0]
        # assert len(batch['gt_objns']) == OBJN_BATCH_SIZE

        if config.USE_CATS:
            batch['cat_filter'] = batch['gt_cats'] > 0
            batch['gt_cats'] = batch['gt_cats'][batch['cat_filter']] - 1

        mask_ids = np.random.choice(len(batch['gt_masks']), min(len(batch['gt_masks']), MASK_BATCH_SIZE), replace=False)
        mask_bool = np.zeros(len(batch['gt_masks']))
        mask_bool[mask_ids] = 1
        mask_filter_ids = np.where(batch['mask_filter'])[0]
        mask_filter_ids = mask_filter_ids[mask_ids]
        batch['mask_filter'][...] = 0
        batch['mask_filter'][mask_filter_ids] = 1
        batch['gt_masks'] = batch['gt_masks'][mask_bool == 1]
        batch['gt_atts'] = batch['gt_atts'][mask_bool == 1]

        return batch



class COCOSSMDemoSpider(BaseCOCOSSMSpider):

    SCALE = 830
    random_idx = False
    LAYER_NUM = 4
    RFs = [16, 24, 32, 48, 64, 96, 128]

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, False)
            # 0 for '__background__'
            self.__class__.cats_to_labels = dict([(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMDemoSpider, self).__init__(*args, **kwargs)
        try:
            self.RFs = RFs
        except Exception as e:
            pass
        try:
            self.SCALE = TEST_SCALE
        except Exception as e:
            pass


    def fetch(self):
        idx = self.get_idx()
        self.flipped = False
        item = self.dataset[idx]
        self.image_path = item.image_path
        self.anns = item.imgToAnns
        self.max_edge = self.SCALE
        self.fetch_image()
        return {"image": self.img_blob}

    def next_idx(self):
        self._idx += 1
