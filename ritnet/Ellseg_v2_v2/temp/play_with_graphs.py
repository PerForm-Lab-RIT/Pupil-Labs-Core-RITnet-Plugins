#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 15:33:27 2021

@author: rsk3900
"""

import matplotlib.pyplot as plt
import networkx as nx
import random

net = nx.MultiDiGraph()

list_ds = ['OpenEDS', 'NVGaze', 'UnityEyes', 'riteyes-s-general',
           'riteyes-s-natural', 'LPW', 'Santini', 'Fuhl', 'Swirski']

edges = []
for ds_train in list_ds:
    for ds_test in list_ds:
        if ds_train != ds_test:
            edges.append((ds_train, ds_test, random.random()))

net.add_weighted_edges_from(edges)

# max_edge = max([ele['value'] for ele in net.edges])

# for edge in net.edges:
#     edge['value'] = (max_edge - edge['value'])/max_edge

# net.show_buttons()

fig, axs = plt.subplots()
nx.draw_circular(net, with_labels=True)

