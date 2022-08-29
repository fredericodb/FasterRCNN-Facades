#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/datasets/fix_imagesets.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PASCAL Visual Object Classes dataset loader. Datasets available at:
# http://host.robots.ox.ac.uk/pascal/VOC/
#
# The dataset directory must contain the following sub-directories:
#
#   Annotations/
#   ImageSets/
#   JPEGImages/
#
# Typically, VOC datasets are stored in a VOCdevkit/ directory and identified
# by year (e.g., VOC2007, VOC2012). So, e.g., the VOC2007 dataset directory
# path would be: VOCdevkit/VOC2007
#

from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET


from tqdm import tqdm


def _get_annotations(dir, annot_dir):
    classes_count = {}
    class_mapping = {}

    # pascal voc directory + 2012
    data_paths = dir

    print('Parsing annotation files')
    for data_path in data_paths:

        annot_path = os.path.join(data_path, annot_dir)

        # annotation 파일 read
        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]

        annots = tqdm(annots)
        for annot in annots:
            # try:
            annots.set_description("Processing %s" % annot.split(os.sep)[-1])

            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')

            assert len(element_objs) > 0

            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                # class mapping 정보 추가
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)  # 마지막 번호로 추가

    return classes_count, class_mapping


def _get_classes(split, dir):
    imageset_dir = os.path.join(dir, "ImageSets", "Main")
    classes = set(
        [os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_" + split + ".txt")])
    assert len(classes) > 0, "No classes found in ImageSets/Main for '%s' split" % split
    class_index_to_name = {(1 + v[0]): v[1] for v in enumerate(sorted(classes))}
    class_index_to_name[0] = "background"
    return class_index_to_name


def _get_filepaths(split, dir):
    image_list_file = os.path.join(dir, "ImageSets", "Main", split + ".txt")
    with open(image_list_file) as fp:
        basenames = [line.strip() for line in fp.readlines()]  # strip newlines
    image_paths = [os.path.join(dir, "JPEGImages", basename) + ".jpg" for basename in basenames]
    return image_paths


def _save_filepaths(split, dir, basenames):
    image_list_file = os.path.join(dir, "ImageSets", "Main", split + ".txt")
    with open(image_list_file, 'w') as fp:
        fp.writelines(basename + "\n" for basename in basenames)


def get_good_voc_imagesets(dir, basenames):
    filepaths = [os.path.join(dir, "Annotations", basename) for basename in basenames]
    new_basenames = []
    for filepath in filepaths:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        annotation_file = os.path.join(dir, "Annotations", basename) + ".xml"
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        assert tree != None, "Failed to parse %s" % annotation_file
        assert len(root.findall("size")) == 1
        size = root.find("size")
        assert len(size.findall("depth")) == 1
        depth = int(size.find("depth").text)
        assert depth == 3
        boxes = []
        for obj in root.findall("object"):
            assert len(obj.findall("name")) == 1
            assert len(obj.findall("bndbox")) == 1
            assert len(obj.findall("difficult")) == 1
            if obj.find("difficult").text == 'Unspecified':
                is_difficult = 0
            else:
                is_difficult = int(obj.find("difficult").text) != 0
            class_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            assert len(bndbox.findall("xmin")) == 1
            assert len(bndbox.findall("ymin")) == 1
            assert len(bndbox.findall("xmax")) == 1
            assert len(bndbox.findall("ymax")) == 1
            x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
            y_min = int(bndbox.find("ymin").text) - 1
            x_max = int(bndbox.find("xmax").text) - 1
            y_max = int(bndbox.find("ymax").text) - 1
            corners = np.array([y_min, x_min, y_max, x_max]).astype(np.float32)
        if len(root.findall("object")) > 0:
            new_basenames.append(basename)
        else:
            print('Example %s deleted.' % basename)
        # assert len(boxes) > 0, "Image without object %s" % annotation_file
    return new_basenames


def fix_voc_imagesets(split, dir):
    filepaths = _get_filepaths(split, dir)
    basenames = []
    for filepath in filepaths:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        annotation_file = os.path.join(dir, "Annotations", basename) + ".xml"
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        assert tree != None, "Failed to parse %s" % annotation_file
        assert len(root.findall("size")) == 1
        size = root.find("size")
        assert len(size.findall("depth")) == 1
        depth = int(size.find("depth").text)
        assert depth == 3
        boxes = []
        for obj in root.findall("object"):
            assert len(obj.findall("name")) == 1
            assert len(obj.findall("bndbox")) == 1
            assert len(obj.findall("difficult")) == 1
            if obj.find("difficult").text == 'Unspecified':
                is_difficult = 0
            else:
                is_difficult = int(obj.find("difficult").text) != 0
            class_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            assert len(bndbox.findall("xmin")) == 1
            assert len(bndbox.findall("ymin")) == 1
            assert len(bndbox.findall("xmax")) == 1
            assert len(bndbox.findall("ymax")) == 1
            x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
            y_min = int(bndbox.find("ymin").text) - 1
            x_max = int(bndbox.find("xmax").text) - 1
            y_max = int(bndbox.find("ymax").text) - 1
            corners = np.array([y_min, x_min, y_max, x_max]).astype(np.float32)
        if len(root.findall("object")) > 0:
            basenames.append(basename)
        else:
            print('Example %s deleted.' % basename)
        # assert len(boxes) > 0, "Image without object %s" % annotation_file
    return basenames


def get_all_imagenames(dir):
    imageset_dir = os.path.join(dir, "JPEGImages")
    basenames = set(
        [os.path.splitext(os.path.basename(path))[0] for path in Path(imageset_dir).glob("*.jpg")])
    return basenames


def get_all_basenames(dir):
    imageset_dir = os.path.join(dir, "Annotations")
    basenames = set(
        [os.path.splitext(os.path.basename(path))[0] for path in Path(imageset_dir).glob("*.xml")])
    return basenames


def look_for_imagenames(dir, basenames):
    imageset_dir = os.path.join(dir, "JPEGImages")
    new_basenames = []
    for filepath in Path(imageset_dir).glob("*.jpg"):
        imagename = os.path.splitext(os.path.basename(filepath))[0]
        if imagename in basenames:
            new_basenames.append(imagename)
    return new_basenames


dir = '/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha_plus'
basenames = get_all_basenames(dir)
real_basenames = look_for_imagenames(dir, basenames)
print("%d basenames deleted." % (len(basenames) - len(real_basenames)))
print("%d real basenames (image+annot)." % len(real_basenames))
good_basenames = get_good_voc_imagesets(dir, basenames)
print("%d real basenames deleted." % (len(real_basenames) - len(good_basenames)))
print("%d good basenames with objects." % len(good_basenames))
random.shuffle(good_basenames)
n = len(good_basenames)

# If the size of the dataset is 100 to 10K ~ 60/20/20
# If the size of the dataset is 1M to INF ==> 98/1/1 or 99.5/0.25/0.25
p_wtrain = 0.50
n_wtrain = int(p_wtrain * n)
wtrain = good_basenames[:n_wtrain]
good_basenames = good_basenames[n_wtrain:]
print("%d examples for semi-supervised training" % n_wtrain)

n = n - n_wtrain

if n <= 10000:
    p_train = 0.60
    p_val = 0.20
    p_test = 0.20
elif n >= 10e6:
    p_train = 0.98
    p_val = 0.01
    p_test = 0.01

n_train = int(p_train * n)
n_val = int(p_val * n)
n_test = int(p_test * n)
print("%d examples for training" % n_train)
print("%d examples for validation" % n_val)
print("%d examples for testing" % n_test)
print("%d examples total" % n)

i_train = 0
f_train = i_train + n_train
i_val = f_train
f_val = i_val + n_val
i_test = f_val
f_test = n

train = good_basenames[i_train:f_train]
val = good_basenames[i_val:f_val]
test = good_basenames[i_test:f_test]

_save_filepaths('train', dir, train)
_save_filepaths('val', dir, val)
_save_filepaths('test', dir, test)

imagenames = get_all_imagenames(dir)
print("%d images" % len(imagenames))
