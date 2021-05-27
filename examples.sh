#!/bin/bash


python run.py --do_train --do_valid --do_test -lr 0.0005 --warm_up_steps 10000 --max_steps 500000 \
-de -dr -g 500 -r 0.00000002 -a 1.0 -adv --model QuatE -d 1000 --batch_size 1024 -n 128 \
--print_on_screen --cuda --valid_steps 5000 --test_batch_size 256 --save_checkpoint_steps 10000