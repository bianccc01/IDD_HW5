CUDA_VISIBLE_DEVICES=0 python3 matcher.py \
  --task wdc_all_small \
  --input_path data/input/input.jsonl \
  --output_path data/output/output.jsonl \
  --lm distilbert \
  --max_len 64 \
  --use_gpu \
  --fp16 \
  --checkpoint_path checkpoints/