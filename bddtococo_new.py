import os
import argparse
import json
import sys
import math
from Bezier import bezier_curve

line_dict = {
    "double white solid" : 0,
    "double white dashed" : 1,
    "double yellow solid" : 2,
    "double yellow dashed" : 3,
    "double other solid" : 4,
    "double other dashed" : 5,
    "single white solid" : 6,
    "single white dashed" : 7,
    "single yellow solid" : 8,
    "single yellow dashed" : 9,
    "single other solid" : 10,
    "single other dashed" : 11,
}
ignore_line =["lane/crosswalk","lane/road curb"]

def make_coco_categories(src):
    # print("making coco categories_info...")
#     categoriesList = []
    if not categoriesList:
        for i in src["frames"][0]["objects"]:
            eachcategoryDict = {}
            info=i["category"]
            if (info[:5]=="lane/") and (info not in ignore_line) and (i["attributes"]["direction"]=="parallel"):
                line_name=info[5:]+' '+i["attributes"]["style"]
                if line_dict[line_name] not in exit_id:
                    exit_id.append(line_dict[line_name])
                    eachcategoryDict['supercategory'] = line_name
                    eachcategoryDict['name'] = line_name
                    eachcategoryDict['id']=line_dict[line_name]
                    categoriesList.append(eachcategoryDict)
            categoriesList.sort(key=lambda keys: keys.get("id"))
    return categoriesList

def make_coco_images(src):
    # print("making coco images_info...")
    imagesList=[]
    if src["name"] not in exit_image:
        imageDict={}
        imageDict["height"]=720
        imageDict["width"]=1280
        imageDict["id"]=image_id
        imageDict["file_name"]=src["name"]+'.jpg'
        exit_image.append(src["name"])
        imagesList.append(imageDict)
    return imagesList

def maybe_same_lane(cur_lane, another_lane):  #判断是否可能框出同一车道线(根据种类和线虚实)
    if ( cur_lane["category"] == another_lane["category"] and
            cur_lane["attributes"] == another_lane["attributes"] ):
        return True
    else:
        return False

def select_line_list(src):   #选择出平行且符合待检测要求的外围线
    line_list=[]
    for i in src["frames"][0]["objects"]:
        info=i["category"]
        if (info[:5]=="lane/") and (info not in ignore_line) and (i["attributes"]["direction"]=="parallel"):
            line_list.append(i)
    return line_list

def transform_bezier(src,GRID_NUM):
    line=select_line_list(src)
    for i in line:
        lane=i["poly2d"]
        # print(lane)
        if len(lane)==4:
            xvals, yvals = bezier_curve(lane, nTimes=GRID_NUM)
#             i["poly2d"] = [[xvals[i], yvals[i], "C"] for i in range(len(xvals))]
            i["poly2d"] = [[xvals[i], yvals[i]] for i in range(len(xvals))]
        i["poly2d"].sort(key=lambda x:x[1])

def dist(pset1, pset2):
    set2_down_pt = pset2[-1]
    set2_up_pt = pset2[0]
    set1_down_pt = pset1[-1]
    set1_up_pt = pset1[0]
    
    dis1 = math.sqrt((set1_down_pt[0] - set2_down_pt[0])**2 + (set1_down_pt[1] - set2_down_pt[1])**2)
    dis2 = math.sqrt((set1_up_pt[0] - set2_up_pt[0])**2 + (set1_up_pt[1] - set2_up_pt[1])**2)
    max_dis = max(dis1, dis2)
    return max_dis

def find_line_couple(src):
    line_couple=[]
    line=select_line_list(src)
    for i in range(len(line)-1):
        if maybe_same_lane(line[i],line[i+1]):
            if dist(line[i]["poly2d"],line[i+1]["poly2d"])<200:
                line_couple.append([line[i],line[i+1]])
                if i+2<len(line):
                    i+=2
                else:
                    break
    return line_couple

def make_coco_annotation(src,coco_images):
    # print("making coco annotation_info...")
    annotationsList=[]
    global annotation_id
    if not annotationsList:
        line_couple=find_line_couple(src)
        # print(line_couple)
        for i in line_couple:
            eachannotationDict={}
            lane1=i[0]["poly2d"]
            lane2=i[1]["poly2d"]
            # print(id_num)
            # print(len(lane1))
            # print(len(lane2))
            if len(lane1)==2 and len(lane2)==2:
                eachannotationDict["segmentation"]=[[lane1[0][0],lane1[0][1],lane1[1][0],lane1[1][1],lane2[1][0],lane2[1][1],lane2[0][0],lane2[0][1]]]
                line_name=i[0]["category"][5:]+' '+i[0]["attributes"]["style"]
                eachannotationDict["category_id"]=line_dict[line_name]
                eachannotationDict["iscrowd"]=0
                eachannotationDict["image_id"]=coco_images[0]["id"]
                eachannotationDict["id"]=annotation_id
                annotation_id+=1
                eachannotationDict["area"]=0
                
                min_x=int(min(eachannotationDict["segmentation"][0][::2]))
                min_y=int(min(eachannotationDict["segmentation"][0][1::2]))
                max_x=int(max(eachannotationDict["segmentation"][0][::2]))
                max_y=int(max(eachannotationDict["segmentation"][0][1::2]))
                max_width=max_x-min_x
                max_depth=max_y-min_y
                
                eachannotationDict["bbox"]=[min_x,min_y,max_width,max_depth]
                annotationsList.append(eachannotationDict)
            if len(lane1)>10 and len(lane2)==2:
                lane1_new=[n for a in lane1 for n in a]
                eachannotationDict["segmentation"]=[lane1_new+[lane2[1][0],lane2[1][1],lane2[0][0],lane2[0][1]]]
                line_name=i[0]["category"][5:]+' '+i[0]["attributes"]["style"]
                eachannotationDict["category_id"]=line_dict[line_name]
                eachannotationDict["iscrowd"]=0
                eachannotationDict["image_id"]=coco_images[0]["id"]
                eachannotationDict["id"]=annotation_id
                annotation_id+=1
                eachannotationDict["area"]=0
                
                min_x=int(min(eachannotationDict["segmentation"][0][::2]))
                min_y=int(min(eachannotationDict["segmentation"][0][1::2]))
                max_x=int(max(eachannotationDict["segmentation"][0][::2]))
                max_y=int(max(eachannotationDict["segmentation"][0][1::2]))
                max_width=max_x-min_x
                max_depth=max_y-min_y
                
                eachannotationDict["bbox"]=[min_x,min_y,max_width,max_depth]
                annotationsList.append(eachannotationDict)
            if len(lane1)==2 and len(lane2)>10:
                lane2.reverse()
                lane2_new=[n for a in lane2 for n in a]
                eachannotationDict["segmentation"]=[[lane1[0][0],lane1[0][1],lane1[1][0],lane1[1][1]]+lane2_new]
                line_name=i[0]["category"][5:]+' '+i[0]["attributes"]["style"]
                eachannotationDict["category_id"]=line_dict[line_name]
                eachannotationDict["iscrowd"]=0
                eachannotationDict["image_id"]=coco_images[0]["id"]
                eachannotationDict["id"]=annotation_id
                annotation_id+=1
                eachannotationDict["area"]=0
                
                # print(eachannotationDict["segmentation"][0])
                min_x=int(min(eachannotationDict["segmentation"][0][::2]))
                min_y=int(min(eachannotationDict["segmentation"][0][1::2]))
                max_x=int(max(eachannotationDict["segmentation"][0][::2]))
                max_y=int(max(eachannotationDict["segmentation"][0][1::2]))
                max_width=max_x-min_x
                max_depth=max_y-min_y
                
                eachannotationDict["bbox"]=[min_x,min_y,max_width,max_depth]
                annotationsList.append(eachannotationDict)
                
            if len(lane1)==3 and len(lane2)==3:
                eachannotationDict["segmentation"]=[[lane1[0][0],lane1[0][1],lane1[1][0],lane1[1][1],lane1[2][0],lane1[2][1],lane2[2][0],lane2[2][1],lane2[1][0],lane2[1][1],lane2[0][0],lane2[0][1]]]
                line_name=i[0]["category"][5:]+' '+i[0]["attributes"]["style"]
                eachannotationDict["category_id"]=line_dict[line_name]
                eachannotationDict["iscrowd"]=0
                eachannotationDict["image_id"]=coco_images[0]["id"]
                eachannotationDict["id"]=annotation_id
                annotation_id+=1
                eachannotationDict["area"]=0
                
                # print(eachannotationDict["segmentation"][0])
                min_x=int(min(eachannotationDict["segmentation"][0][::2]))
                min_y=int(min(eachannotationDict["segmentation"][0][1::2]))
                max_x=int(max(eachannotationDict["segmentation"][0][::2]))
                max_y=int(max(eachannotationDict["segmentation"][0][1::2]))
                max_width=max_x-min_x
                max_depth=max_y-min_y
                
                eachannotationDict["bbox"]=[min_x,min_y,max_width,max_depth]
                annotationsList.append(eachannotationDict)
                
            if len(lane1)>10 and len(lane2)>10:
                lane1_new=[n for a in lane1 for n in a]
                lane2.reverse()
                lane2_new=[n for a in lane2 for n in a]
                eachannotationDict["segmentation"]=[lane1_new+lane2_new]
                line_name=i[0]["category"][5:]+' '+i[0]["attributes"]["style"]
                eachannotationDict["category_id"]=line_dict[line_name]
                eachannotationDict["iscrowd"]=0
                eachannotationDict["image_id"]=coco_images[0]["id"]
                eachannotationDict["id"]=annotation_id
                annotation_id+=1
                eachannotationDict["area"]=0
                
                min_x=int(min(eachannotationDict["segmentation"][0][::2]))
                min_y=int(min(eachannotationDict["segmentation"][0][1::2]))
                max_x=int(max(eachannotationDict["segmentation"][0][::2]))
                max_y=int(max(eachannotationDict["segmentation"][0][1::2]))
                max_width=max_x-min_x
                max_depth=max_y-min_y
                
                eachannotationDict["bbox"]=[min_x,min_y,max_width,max_depth]
                annotationsList.append(eachannotationDict)
                
    return annotationsList

def tranform_one_json(json_path,GRID_NUM):
    one_json_src = json.load(open(json_path))
    global image_id
    d={}
    coco_images=make_coco_images(one_json_src)
    image_id+=1
    d["images"]=coco_images
    d["categories"]=make_coco_categories(one_json_src)
    transform_bezier(one_json_src,GRID_NUM)
    d["annotations"]=make_coco_annotation(one_json_src,coco_images)
    # print(d["images"][0]["file_name"])
    save_path="coco_val_json/"
    file_name=json_path.lstrip("labels/100k/train/")
    # file_name=str(num)+".json"
    save_file=save_path+file_name
    with open(save_file,'w') as file_obj:
        json.dump(d,file_obj)

if __name__ == "__main__":
    image_id=0
    annotation_id=0
    categoriesList = []
    exit_id=[]
# imagesList=[]
    exit_image=[]
# annotationsList=[]
    GRID_NUM=100
    src_path="labels/100k/train/"

    for root, dirs, files in os.walk(src_path):
        for file in files:
            if ".json" in file:
                tranform_one_json(src_path+file,GRID_NUM)
                categoriesList = []
                exit_id=[]