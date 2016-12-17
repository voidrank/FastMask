import caffe
import os

import config
import numpy as np
from spiders.coco_ssm_spider import * 
from alchemy.engines.caffe_python_layers import AlchemyDataLayer


class COCOSSMDataLayer(AlchemyDataLayer):

    spider = COCOSSMSpider
    max_cache_item_num = 500
    wait_time = 10
    process_num = config.DATA_PROVIDER_PROCESS

