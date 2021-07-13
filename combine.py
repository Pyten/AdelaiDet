import os
import datetime
import random
import sys
import operator
import math
import numpy as np
# import skimage.io
# import matplotlib
# from matplotlib import pyplot as plt
# import cv2
from collections import defaultdict, OrderedDict
import json
#待合并的路径
FILE_DIR = "coco_val_json/"
 
def load_json(filenamejson):
    with open(filenamejson) as f:
        raw_data = json.load(f)
    return raw_data
 
file_count = 0
files = next(os.walk(FILE_DIR))[2]
for x in range(len(files)):
    #解析文件名字和后缀
    #print(str(files[x]))
    file_suffix = str(files[x]).split(".")[1]
    file_name = str(files[x]).split(".")[0]
#     #过滤类型不对的文件
    if file_suffix != "json":
        continue
    #计数
    file_count = file_count + 1
    #组合文件路径
    filenamejson = FILE_DIR + str(files[x])
    #读取文件
    if file_count == 1:
        #第一个文件作为root
        root_data = load_json(filenamejson)
    else:
        raw_data = load_json(filenamejson)
        #追加images的数据
        root_data['images'].append(raw_data['images'][0])
        #统计区域个数
        annotation_count = str(raw_data["annotations"]).count('image_id',0,len(str(raw_data["annotations"])))
        for i in range(annotation_count):
            #追加annotations的数据
            root_data['annotations'].append(raw_data['annotations'][i])
        #统计根文件类别数
        #root_categories_count = str(root_data["categories"]).count('name',0,len(str(root_data["categories"])))
        #统计这个文件的类别数
        raw_categories_count = str(raw_data["categories"]).count('name',0,len(str(raw_data["categories"])))
        for j in range(raw_categories_count):
            root_data["categories"].append(raw_data['categories'][j])
temp = []
for m in root_data["categories"]:
    if m not in temp:
        temp.append(m)
root_data["categories"] = temp
print("共处理 {0} 个json文件".format(file_count))
print("共找到 {0} 个类别".format(str(root_data["categories"]).count('name',0,len(str(root_data["categories"])))))
 
json_str = json.dumps(root_data)
with open('test.json', 'w') as json_file:
    json_file.write(json_str)
#写出合并文件
 
print("Done!") 