#!/bin/bash -l

command_1="python play_with_bash.py --num 1"
command_2="python play_with_bash.py --num 2"
command_3="python play_with_bash.py --num 3"

eval "(${command_1}; ${command_2}) & ${command_3}$"