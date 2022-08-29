#!/bin/sh
python -m tf2.FasterRCNN --predict-all=test --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha_plus --backbone=vgg16 --load-from=fasterrcnn_tf2.h5
