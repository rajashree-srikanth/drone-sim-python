#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi


class WindField:
    def __init__(self, w=[0.,0.]):
        self.w = w
        
    def sample_num(self, _x, _y, _t):
        return self.w

    def sample_sym(self, _x, _y, _t):
        return self.w
    
