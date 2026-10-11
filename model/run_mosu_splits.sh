# -*- coding: utf-8 -*-
# Bash script to automate the training of the model for the `MoSu` subsample (see tools/make_mosu_subset.py).
# Runs main.py, for each data-split and valid values of `reg_factor`, for the selected number of epochs and batch size.
# MoSu features are 768-d CLIP vectors (one per second), hence --input_size 768.

LC_NUMERIC="en_US.UTF-8"
for i in $(seq 0 4); do
  for sigma in $(seq 0.5 0.1 0.9); do
    python model/main.py --split_index "$i" --n_epochs 400 --batch_size 40 --video_type 'MoSu' --input_size 768 --reg_factor "$sigma"
  done
done
