# -*- coding:utf-8 -*-

from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
# import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json

json_file='../data/mscoco2014/annotations/person_keypoints_val2014.json' # # Object Instance 类型的标注
# person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
# captions_val2017.json  # Image Caption的标注格式

data=json.load(open(json_file,'r'))

data_2={}
data_2['info']=data['info']
data_2['licenses']=data['licenses']
data_2['images']=[data['images'][0]] # 只提取第一张图片
data_2['categories']=data['categories']
annotation=[]

# 通过imgID 找到其所有对象
imgID=data_2['images'][0]['id']
for ann in data['annotations']:
    if ann['image_id']==imgID:
        annotation.append(ann)

data_2['annotations']=annotation

# 保存到新的JSON文件，便于查看数据特点
json.dump(data_2,open('./pickOne_person_keypoints_2014.json','w'),indent=4) # indent=4 更加美观显示