# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:27:44 2021

@author: Rudra
"""
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num', default=0)

args = parser.parse_args()

time.sleep(10)

print('The number is: {}'.format(args.num))
