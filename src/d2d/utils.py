#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

