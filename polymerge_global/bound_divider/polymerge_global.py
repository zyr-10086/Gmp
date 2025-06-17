import sys

from ..map_stitch_libs import post_process_lib as pplb
from ..map_stitch_libs import Stitch_lib as slb
from ..map_stitch_libs import PolyMergelib as plb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString

class PolyMerge_Global:
    def __init__(self):
        self.stitch_data = None

    def preprocess_data(self,data):
        global_map = [[], [], []]
    
        car_trajectory = []
        # data = data[:10]
        for index in range(0,len(data)):
            frame = data[index]
            loc = frame['gt']['pose']
            yaw = frame['gt']['yaw'] - np.pi / 2
            car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
            message = slb.preprocess_frame(frame)        
            slb.update_global_map(global_map, message, type='pred')
        
        return global_map, car_trajectory
    
    def format_result(self, refined_global_map):
        pred_datas = {'divider': [], 'ped_crossing': [], 'boundary': []}
        id2cat = ['divider', 'ped_crossing', 'boundary']
        simplify = 0.5
        for category in range(3):
            if len(refined_global_map[category]) == 0:
                continue
            for frame_vec in refined_global_map[category]:
                for vec in frame_vec:
                    polyline = LineString(vec)
                    polyline = np.array(polyline.simplify(simplify).coords)
                    pred_datas[id2cat[category]].append(vec)            
        
        return pred_datas
    
    def vis_maps(car_trajectory, refined_global_map, pred_save_path=None):
        all_points = []
        for catogory in refined_global_map:
            for frame in catogory:
                for vec in frame:
                    all_points.append(vec)
        all_points = np.concatenate(all_points, axis=0)
        x_min = all_points[:,0].min()
        x_max = all_points[:,0].max()
        y_min = all_points[:,1].min()
        y_max = all_points[:,1].max()
        plb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, refined_global_map)

    def global_map_stitch(self,global_data, if_vis=False, pred_save_path=None):
        global_map, car_trajectory = self.preprocess_data(global_data)
        refined_global_map = plb.refine_token_v2(global_map, proximity_th = 0.8)
        merge_result = self.format_result(refined_global_map)
        self.stitch_data = merge_result

        if if_vis:
            self.vis_maps(car_trajectory, refined_global_map, pred_save_path)
            
        return merge_result