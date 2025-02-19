#!/bin/bash

# Controlla se una GPU è disponibile
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  GPU non disponibile, verrà utilizzata la CPU."
    GPU_FLAG=""
else
    GPU_FLAG="--use_gpu --fp16"
fi

# Avvia la predizione con Ditto
CUDA_VISIBLE_DEVICES=0 python3 matcher.py \
  --task companies \
  --input_path data/input/ditto_input.jsonl \
  --output_path data/output/ditto_predictions.jsonl \
  --lm roberta-base \
  --max_len 64 \
  $GPU_FLAG \
  --checkpoint_path checkpoints/

echo "✅ Predizione completata. Output salvato in data/output/ditto_predictions.jsonl"

# parser for jsonl to csv
python3 parser_result.py --jsonl data/output/ditto_predictions.jsonl --csv data/output/ditto_predictions(NO-AD).csv --val_csv data/output/ditto_val.csv

echo "✅ Conversione in CSV completata. Output salvato in data/output/ditto_predictions.csv"
