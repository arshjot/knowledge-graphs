#!/bin/bash


python run.py --do_train --do_valid --do-test -lr 0.001 --warm_up_steps 100000 --max_steps 500000 \
-de -dr -g 500 -r 0.000002 --model ConvE -d 1000 --batch_size 128 -acc 4 -n 128 --print_on_screen \
--cuda --valid_steps 2000 --test_batch_size 64 --save_checkpoint_steps 10000