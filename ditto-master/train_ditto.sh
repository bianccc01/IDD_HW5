python3 train_ditto.py \
  --task companies \
  --batch_size 64 \
  --max_len 64 \
  --lr 3e-5 \
  --n_epochs 40 \
  --lm distilbert \
  --da del \
  --dk product \
  --summarize