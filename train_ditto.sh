#!/bin/bash

python3 ditto-master/train_ditto.py \
  --train data/record_linkage/ditto/train/ditto_train.txt \
  --dev data/ditto_dev.txt \
  --test data/ditto_test.txt \
  --model roberta \
  --batch_size 16 \
  --max_len 256 \
  --lr 3e-5 \
  --epochs 5 \
  --output_model ditto_model.pt \
  --fp16
