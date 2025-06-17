import sys
sys.path.append('..')
from map_stitch_libs import post_process_lib as pplb
from map_stitch_libs import Stitch_lib as slb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev
from progress.bar import Bar
import time

if __name__ == '__main__':

    data_path = 'match_ss.pkl'
    with open(data_path, 'rb') as f:
        match_result = pickle.load(f)
    merged_global = [[],[],[]]
    start_time = time.time()
    for cate in range(3):
        merged_frame = []
        match_result_by_cate= match_result[cate]
        bar = Bar('merging', max=len(match_result_by_cate))
        for instance_set in match_result_by_cate:
            result = [[],[]]
            merged_poly, left, right = slb.merge_recursive(instance_set, result, cate)
            merged_frame.append(merged_poly)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'merging: {elapsed_time:.2f} s'
        bar.finish()

        merged_global[cate].append(merged_frame)
    # import ipdb;ipdb.set_trace()
    all_points = []

    for catogory in merged_global:
        for frame in catogory:
            for vec in frame:
                all_points.append(vec)
    # import ipdb;ipdb.set_trace()

    all_points = np.concatenate(all_points, axis=0)
    x_min = all_points[:,0].min()
    x_max = all_points[:,0].max()
    y_min = all_points[:,1].min()
    y_max = all_points[:,1].max()


    pred_save_path = 'pred_merge_global_polymerge_scene2.png'
    slb.plot_fig_merged( [], x_min, x_max, y_min, y_max, pred_save_path, merged_global)