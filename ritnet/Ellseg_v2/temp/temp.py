#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 01:42:14 2021

@author: rsk3900
"""

import matplotlib.pyplot as plt
import numpy as np

A = 5*np.random.rand(100, 1)
B = 10*np.random.rand(100, 1) -5
C = 7*np.random.rand(100, 1) - 5

fig, axs = plt.subplots()
bp0 = axs.boxplot(A, positions=[0], patch_artist=True)
bp1 = axs.boxplot(B, positions=[1], patch_artist=True)
bp2 = axs.boxplot(C, positions=[2], patch_artist=True)

colors = ['pink', 'lightblue', 'darkgreen']
for bplot, color in zip((bp0, bp1, bp2), colors):
    for patch in bplot['boxes']:
        patch.set_facecolor(color)

axs.legend(['A', 'B', 'C'])
