import Stitch_lib as slb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

if __name__ == '__main__':
    data = slb.load_data('datas/merge_divider_boundary.pkl')
    slb.plot_map(data, 'fit_merged', save = True, fit = True)