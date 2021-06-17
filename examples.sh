#!/bin/bash


!python run.py --do_train --do_valid --do_test -lr 0.001 --warm_up_steps 20000 --max_steps 300000 \
-g 500 -r 0.0000002 --model HolE  -a 1.0 -adv -d 2000 --batch_size 512 -acc 2 -n 128 --print_on_screen \
--cuda --valid_steps 10000 --test_batch_size 128 --save_checkpoint_steps 10000

python run.py --do_train --do_valid --do_test -lr 0.001 --warm_up_steps 50000 --max_steps 300000 \
-de -dr -g 500 -r 0.00000002 -a 1.0 -adv --model QuatE -d 1000 --batch_size 1024 -n 128 \
--print_on_screen --cuda --valid_steps 10000 --test_batch_size 256 --save_checkpoint_steps 20000

python run.py --do_train --do_valid -lr 0.0005 --warm_up_steps 10000 --max_steps 500000 -de -dr \
-g 500 -r 0.00000002 -a 1.0 -adv --model OctonionE -d 1000 --batch_size 1024 -n 128 --print_on_screen \
--cuda --valid_steps 10000 --test_batch_size 256 --save_checkpoint_steps 10000