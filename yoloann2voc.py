import os
import numpy as np
import codecs
import pandas as pd
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
from IPython import embed
classname_to_id = {"person": 0, "hat": 1}
#1.标签路径
# csv_file = "../csv/train_labels.csv"
ann_file = '/data/datasets/tianji/hat/train(1).txt'
ori_train_txt = '/data/datasets/tianji/hat/hat_train.txt'
ori_val_txt = '/data/datasets/tianji/hat/hat_val.txt'
f = open(ann_file, 'r')
image_dir = "/data/datasets/tianji/hat/安全帽数据/saftyHat/Saftyhat/"
saved_path = "/data/datasets/tianji/hat/VOCdevkit/VOC2007/"                #保存路径
# image_save_path = "/data/datasets/tianji/hat/JPEGImages/"
# image_raw_parh = "/data/datasets/tianji/hat/csv/images/"
#2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")


def get_key_from_value(value):
    for k, v in classname_to_id.items():
        if v == value:
            return k
    return None


#3.获取待处理文件
total_csv_annotations = {}
# annotations = pd.read_csv(csv_file,header=None).values
for line in f.readlines():
    contents = line.split(' ')
    if len(contents) < 2:
        continue
    key = os.path.basename(contents[0])
    # value = np.array([annotation[1:]])
    value_list = []
    for box in contents[1:]:
        value = list(map(int, box.split(',')))
        value[-1] = get_key_from_value(int(value[-1]))
        value_list.append(np.array(value))
    total_csv_annotations[key] = np.stack(value_list, axis=0)
f.close()

#4.读取标注信息并写入 xml
for filename,label in total_csv_annotations.items():
    #embed()
    height, width, channels = cv2.imread(image_dir + filename).shape
    #embed()
    with codecs.open(saved_path + "Annotations/"+filename.replace(".jpg",".xml"),"w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
        xml.write('\t<filename>' + filename + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The VOC2007 Database</database>\n')
        xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>HuHui</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        if isinstance(label,float):
            ## 空白
            xml.write('</annotation>')
            continue
        for label_detail in label:
            labels = label_detail
            #embed()
            xmin = int(labels[0])
            ymin = int(labels[1])
            xmax = int(labels[2])
            ymax = int(labels[3])
            label_ = labels[-1]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>'+label_+'</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(filename,xmin,ymin,xmax,ymax,labels)
        xml.write('</annotation>')


#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath+'/trainval.txt', 'w')
# ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = glob(saved_path+"./Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
# for file in total_files:
#     ftrainval.write(file + "\n")

# # move images to voc JPEGImages folder
for image in glob(os.path.join(image_dir, "*")):
    shutil.copy(image, saved_path + "JPEGImages/")

# train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
train_files = []
f_train = open(ori_train_txt, 'r')
for line in f_train.readlines():
    name = os.path.basename(os.path.splitext(line.split(' ')[0])[0])
    train_files.append(name)
f_train.close()
val_files = []
f_val = open(ori_val_txt, 'r')
for line in f_val.readlines():
    name = os.path.basename(os.path.splitext(line.split(' ')[0])[0])
    val_files.append(name)
f_val.close()


for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")

# ftrainval.close()
ftrain.close()
fval.close()
#ftest.close()