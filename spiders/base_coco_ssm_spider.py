import numpy as np
import os
import cv2
import config
from config import *

from alchemy.spiders.dataset_spider import DatasetSpider
from alchemy.datasets.coco import COCO_DS

from pycocotools.mask import frPyObjects

from alchemy.utils.image import (load_image, image_to_data, sub_mean,
                        resize_blob, draw_attention, visualize_bbs,
                        visualize_masks)
from alchemy.utils.mask import (decode, area, toBbox, pts_in_bbs, bbs_in_bbs,
                        encode)

class NoLabelException(Exception):
    pass


class BaseCOCOSSMSpider(DatasetSpider):

    def fetch(self):
        raise NotImplementedError
        
    def fetch_image(self):
        # load
        if os.path.exists(self.image_path) is not True:
            raise IOError("File does not exist: %s" % self.image_path)
        img = load_image(self.image_path)
        self.origin_height = img.shape[0]
        self.origin_width = img.shape[1]

        # resize
        if img.shape[0] > img.shape[1]:
            scale = 1.0 * self.max_edge / img.shape[0]
        else:
            scale = 1.0 * self.max_edge / img.shape[1]

        h, w = int(self.origin_height * scale), int(self.origin_width * scale)
        # make sure that w and h
        # could be divisible by 4
        # so that we can get a 
        # predictable sized feature 
        # map from body net.
        h, w = h - h%4, w - w%4
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = sub_mean(img)
        if self.flipped:
            img = img[:, ::-1, :]
        self.height = img.shape[0]
        self.width = img.shape[1]


        self.image = img
        self.img_blob = self.image.transpose((2, 0, 1))
        self.img_blob = self.img_blob[np.newaxis, ...]

        self.scale = scale

        return {'image': self.img_blob}


    def fetch_masks(self):
        rles = []
        cls = []
        for item in self.anns:
            try:
                rles.append(frPyObjects(item['segmentation'], self.origin_height, self.origin_width)[0])
                cls.append(self.cats_to_labels[item['category_id']])
            except Exception as e:
                # ignore crowd
                # print 'ignore crowd', item['iscrowd']
                pass
        self.cls = cls
        self.masks = decode(rles).astype(np.float)
        self.masks = resize_blob(self.masks, self.image.shape[:2])
        if self.flipped:
            self.masks = self.masks[:, :, ::-1]

    def cal_centers_of_masks_and_bbs(self):
        # (num, 4)
        scale = self.scale
        self.bbs = np.array([np.round(item['bbox']) for item in self.anns]) * scale
        self.bbs_hw = self.bbs[:, (3, 2)]
        self.bbs[:, 2:] += self.bbs[:, :2]
        # (h, w) order
        self.bbs = self.bbs[:, (1, 0, 3, 2)]
        # flip
        h, w = self.image.shape[:2]
        if self.flipped:
            self.bbs = self.bbs[:, (0, 3, 2, 1)]
            self.bbs[:, (1, 3)] = w - self.bbs[:, (1, 3)]
        # (num, 2)
        self.centers = np.array(
            ((self.bbs[:, 0] + self.bbs[:, 2])/2.0,
            (self.bbs[:, 1]+ self.bbs[:, 3])/2.0)
        ).transpose((1, 0))

    def find_matched_masks(self):
        # feature map size
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        # win_pts: [h * w, 2]
        win_pts = np.array((np.arange(h*w, dtype=np.int)/w, np.arange(h*w, dtype=np.int)%w))
        win_pts = win_pts.transpose((1, 0)).astype(np.float)
        win_pts *= ratio
        self.win_pts = win_pts
        # objn_win_cens: [h * w, 4]
        # mask_win_cens: [h * w, 4]
        objn_win_cens = np.hstack((win_pts - (SLIDING_WINDOW_SIZE * ratio * OBJN_CENTER_RATIO/ 2.0), win_pts + (SLIDING_WINDOW_SIZE * ratio * OBJN_CENTER_RATIO / 2.0)))
        mask_win_cens = np.hstack((win_pts - (SLIDING_WINDOW_SIZE * ratio * MASK_CENTER_RATIO/ 2.0), win_pts + (SLIDING_WINDOW_SIZE * ratio * MASK_CENTER_RATIO / 2.0)))
        self.objn_win_cens = objn_win_cens
        self.mask_win_cens = mask_win_cens
        
        # origin image
        img = self.image.copy()
        img += RGB_MEAN

        # visualize centers of windows
        '''
        centers = win_cens + SLIDING_WINDOW_SIZE * CENTER_RATIO * ratio / 2
        bbs = centers[:20, :]
        visualize_bbs(img, bbs)
        '''

        # win_bbs: [h * w, 4]
        win_bbs = np.hstack((win_pts - (SLIDING_WINDOW_SIZE * ratio / 2.0), win_pts + (SLIDING_WINDOW_SIZE * ratio / 2.0)))
        self.win_bbs = win_bbs

        # [h * w, label_num)
        self.objn_match = np.ones((h * w, n), np.int) 
        self.mask_match = np.ones((h * w, n), np.int) 

        # condition 1: neither too large nor too small
        for i in range(n):
            self.objn_match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * OBJN_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * OBJN_UPPER_BOUND_RATIO).all()
        for i in range(n):
            self.mask_match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * MASK_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * MASK_UPPER_BOUND_RATIO).all()

        # condition 2: roughly contained
        for i in range(n):
            self.objn_match[:, i] = self.objn_match[:, i] & pts_in_bbs(self.centers[i], win_bbs)
        for i in range(n):
            self.mask_match[:, i] = self.mask_match[:, i] & pts_in_bbs(self.centers[i], win_bbs)

        # condition 3: roughly centered
        for i in range(n):
            self.objn_match[:, i] = self.objn_match[:, i] & pts_in_bbs(self.centers[i], objn_win_cens)
        for i in range(n):
            self.mask_match[:, i] = self.mask_match[:, i] & pts_in_bbs(self.centers[i], mask_win_cens)

        # choose the closest one
        dist = self.objn_match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        self.objn_match[np.arange(h * w), obj_ids] += 1
        self.objn_match[self.objn_match < 2] = 0
        self.objn_match[self.objn_match == 2] = 1

        dist = self.mask_match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        self.mask_match[np.arange(h * w), obj_ids] += 1
        self.mask_match[self.mask_match < 2] = 0
        self.mask_match[self.mask_match == 2] = 1


        # visualize matched sliding window
        '''
        bbs = np.array((np.nonzero(self.objn_match)[0]/w, np.nonzero(self.objn_match)[0]%w)) * ratio
        bbs = np.vstack((bbs-SLIDING_WINDOW_SIZE/2*ratio, bbs+SLIDING_WINDOW_SIZE/2*ratio)).transpose((1, 0))
        visualize_bbs(img, bbs[:16])
        visualize_bbs(img, bbs[16:])
        '''

        # get hard training samples
        self.get_zoom_negtive_samples()
        self.get_shift_negtive_samples()


    def get_shift_negtive_samples(self):
        # feature map size
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        match = np.ones((h * w, n), np.int)
        win_bbs = self.win_bbs
        objn_win_cens = self.objn_win_cens
        win_pts = self.win_pts
        # condition 1: not too large or too small
        for i in range(n):
            match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * OBJN_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * OBJN_UPPER_BOUND_RATIO).all()

        # condition 2: roughly contained
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], win_bbs)

        '''
        # condition 3: roughly centered
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], objn_win_cens)
        '''
        # choose the closest one
        dist = match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        match[np.arange(h * w), obj_ids] += 1
        match[match < 2] = 0
        match[match == 2] = 1
        self.shift_negtive_match = match

        secondary_objns = np.zeros((h * w))
        secondary_objns[match.any(axis=1)] = 1
        objns = np.zeros((h * w))
        objns[self.objn_match.any(axis=1)] = 1
        negtive_samples = np.zeros((h * w))
        negtive_samples[(secondary_objns == 1) & (objns == 0)] = 1
        if self.shift_negtive_samples is None:
            self.shift_negtive_samples = negtive_samples
        else:
            self.shift_negtive_samples = np.concatenate((self.shift_negtive_samples, negtive_samples), axis=0)


    def get_zoom_negtive_samples(self):
        # feature map size
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        match = np.ones((h * w, n), np.int)
        win_bbs = self.win_bbs
        objn_win_cens = self.objn_win_cens
        win_pts = self.win_pts
        '''
        # condition 1: not too large or too small
        for i in range(n):
            match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * OBJN_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * OBJN_UPPER_BOUND_RATIO).all()
        '''
        # condition 2: roughly contained
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], win_bbs)

        # condition 3: roughly centered
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], objn_win_cens)

        # choose the closest one
        dist = match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        match[np.arange(h * w), obj_ids] += 1
        match[match < 2] = 0
        match[match == 2] = 1
        self.zoom_negtive_match = match

        secondary_objns = np.zeros((h * w))
        secondary_objns[match.any(axis=1)] = 1
        objns = np.zeros((h * w))
        objns[self.objn_match.any(axis=1)] = 1
        negtive_samples = np.zeros((h * w))
        negtive_samples[(secondary_objns == 1) & (objns == 0)] = 1
        if self.zoom_negtive_samples is None:
            self.zoom_negtive_samples = negtive_samples
        else:
            self.zoom_negtive_samples = np.concatenate((self.zoom_negtive_samples, negtive_samples), axis=0)


    def assign_gt(self):
        # feature map size
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        gt_objns = np.ones((h * w))
        gt_objns[np.where(self.objn_match.any(axis=1) == 0)] = 0
        self.gt_objns = gt_objns

        # categories
        if config.USE_CATS:
            gt_cats = np.zeros((h * w))
            labeled_ids = np.where(self.objn_match.any(axis=1) > 0)[0]
            try:
                gt_cats[labeled_ids] = \
                        np.array([self.cls[self.objn_match[i].argmax()] \
                        for i in labeled_ids])
            except Exception as e:
                raise e
            self.gt_cats = gt_cats


        try:
            assert (np.nonzero(self.gt_objns)[0] == np.nonzero(self.objn_match.any(axis=1))[0]).all()
        except Exception as e:
            print np.nonzero(self.gt_objns)[0], np.nonzero(self.objn_match.any(axis=1))[0]
            raise e
        
        # mask_filter
        mask_filter = np.ones((h * w))
        mask_filter[np.where(self.mask_match.any(axis=1) == 0)] = 0
        mask_ids = np.where(mask_filter == 1)[0]
        self.mask_filter = mask_filter

        # masks
        mask_scale = 1.0 / ratio * MASK_SIZE / SLIDING_WINDOW_SIZE
        masks = resize_blob(self.masks, None, mask_scale)
        mh, mw = masks.shape[1:]

        # pad
        pad_masks = np.zeros((n, int(mh+MASK_SIZE*1.5), int(mw+MASK_SIZE*1.5)))
        pad_masks[:, MASK_SIZE/2: mh+MASK_SIZE/2, MASK_SIZE/2: mw+MASK_SIZE/2] = masks
        masks = pad_masks

        # gt masks
        self.gt_masks = np.zeros((h * w, MASK_SIZE, MASK_SIZE))
        # mask_ids = np.where(negtive_samples == 1)[0]
        obj_ids = np.argmax(self.mask_match[mask_ids, :], axis=1)
        i = 0
        scale = MASK_SIZE / SLIDING_WINDOW_SIZE
        for idx in mask_ids:
            self.gt_masks[idx, :, :] = masks[obj_ids[i], idx/w*scale: idx/w*scale+MASK_SIZE, idx%w*scale: idx%w*scale+MASK_SIZE]
            mask = masks[obj_ids[i]]
            i += 1
            # visualization
            '''
            img = cv2.resize(self.image.copy(), None, None, fx=mask_scale, fy=mask_scale, interpolation=cv2.INTER_LINEAR)
            img = img[idx/w*scale-MASK_SIZE/2: idx/w*scale+MASK_SIZE/2, idx%w*scale-MASK_SIZE/2: idx%w*scale+MASK_SIZE/2, :] + RGB_MEAN
            try:
                visualize_masks(img, self.gt_masks[idx:idx+1])
            except Exception :
                pass
                print 'label lies on the edge'
            '''

        # gt attention:
        masks = np.zeros((n, h * scale + MASK_SIZE, w * scale + MASK_SIZE))
        bbs = self.bbs.copy()
        bbs *= mask_scale
        bbs[:, :2] = np.floor(bbs[:, :2])
        bbs[:, 2:] = np.ceil(bbs[:, 2:]).astype(np.int)
        for i in range(n):
            # (h, w) order
            masks[i, bbs[i, 0]+MASK_SIZE/2: bbs[i, 2] + MASK_SIZE/2, bbs[i, 1] + MASK_SIZE/2:bbs[i, 3] + MASK_SIZE/2] = 1
        masks = resize_blob(masks, None, 1.0/scale)
        self.gt_atts = np.zeros((h * w, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_SIZE))
        _ = 0
        for idx in mask_ids:
            i = obj_ids[_]
            x = idx/w
            y = idx%w
            try:
                self.gt_atts[idx, :, :] = masks[i, x: x+SLIDING_WINDOW_SIZE, y: y+SLIDING_WINDOW_SIZE]
            except Exception as e:
                raise e
            _ += 1
            # visualize attetion masks
            '''
            img = self.image[(x-SLIDING_WINDOW_SIZE/2)*ratio: (x+SLIDING_WINDOW_SIZE/2)*ratio, (y-SLIDING_WINDOW_SIZE/2)*ratio: (y+SLIDING_WINDOW_SIZE/2)*ratio, :] + RGB_MEAN
            mask = self.gt_atts[idx: idx+1].copy()
            mask = resize_blob(mask, None, ratio)
            try:
                visualize_masks(img, mask)
            except Exception :
                pass
                print 'label lies on the edge'
            '''


