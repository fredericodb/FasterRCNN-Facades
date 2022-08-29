#!/bin/sh
python -m pytorch.FasterRCNN --predict-all=test --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha --backbone=vgg19 --load-from=fasterrcnn_pytorch.pth
