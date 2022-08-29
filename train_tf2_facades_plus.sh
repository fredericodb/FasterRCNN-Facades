#!/bin/sh
python -m tf2.FasterRCNN --train --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha_plus --annot-dir=ObjectAnnotations --backbone=vgg16 --epochs=10 --learning-rate=1e-3 --save-best-to=fasterrcnn_tf2_tmp.h5
python -m tf2.FasterRCNN --train --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha_plus --annot-dir=ObjectAnnotations --backbone=vgg16 --epochs=4 --learning-rate=1e-4 --load-from=fasterrcnn_tf2_tmp.h5 --save-best-to=fasterrcnn_tf2.h5
rm fasterrcnn_tf2_tmp.h5
