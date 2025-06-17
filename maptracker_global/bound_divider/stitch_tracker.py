import sys
sys.path.append('..')
from map_stitch_libs_global  import Stitch_lib as slb
from map_stitch_libs_global  import Maptracker_lib as mlb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev

if __name__ == '__main__':

    data_path = '../../../local_data/nuscenes/result_keep_scene-0077.pkl'
    matching = 1
    merging = 1

    # matching stage
    if matching:
        data = slb.load_data(data_path)
        # import ipdb;ipdb.set_trace()
        # data = data[:600]
        match_seq, pred_list = mlb.get_scene_matching_result(data['local'])
        match_result = mlb.generate_results(match_seq, pred_list)
        # import ipdb;ipdb.set_trace()
        match_path = 'match_r1.pkl'
        with open(match_path, 'wb') as f:
            pickle.dump(match_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # merging stage
    if merging:
        match_path ='match_r1.pkl'
        with open(match_path, 'rb') as f:
            match_result = pickle.load(f)

        merge_result = mlb.vis_pred_data(match_result, if_vis=True, pred_save_path='pred_merge_result.png')
        # import ipdb;ipdb.set_trace()