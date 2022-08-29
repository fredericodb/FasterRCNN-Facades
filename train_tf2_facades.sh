#!/bin/sh
for a in vgg16
do
  for i in {1..10}
  do
    python -m tf2.FasterRCNN --train --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha --backbone=${a} --epochs=10 --learning-rate=1e-3 --load-from=imagenet --save-best-to=fasterrcnn_tf2_tmp.h5 --log-csv="logs/train_${a}_${i}_metrics.log"
    python -m tf2.FasterRCNN --train --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha --backbone=${a} --epochs=4 --learning-rate=1e-4 --load-from=fasterrcnn_tf2_tmp.h5 --save-best-to=fasterrcnn_tf2.h5 --run-number=${i} --results-to="train_${a}.txt" --log-csv="logs/train_${a}_${i}_metrics2.log"
    rm fasterrcnn_tf2_tmp.h5
    python -m tf2.FasterRCNN --self-train --dataset-dir=/home/fredericobortoloti/Documentos/bigdata/data/gsv_vila_velha --backbone=${a} --epochs=10 --learning-rate=1e-4 --load-from=fasterrcnn_tf2.h5 --save-best-to=fasterrcnn_tf2_ss.h5 --run-number=${i} --results-to="wtrain_${a}.txt" --log-csv="logs/wtrain_${a}_${i}_metrics.log"
  done
done