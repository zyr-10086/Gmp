import sys

from ..map_stitch_libs import post_process_lib as pplb
from ..map_stitch_libs import Stitch_lib as slb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
from progress.bar import Bar
import time

class Ours_Global:
    def __init__(self):
        self.stitch_data = None

    def preprocess_data(self,data):
        global_map = [[], [], []]
    
        car_trajectory = []
        skip_index = 1
        # data = data[:10]
        for index in range(0,len(data)//skip_index):
            frame = data[index * skip_index]
            loc = frame['gt']['pose']
            yaw = frame['gt']['yaw'] - np.pi / 2
            # yaw = frame['gt']['yaw'] - np.pi
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
    
    def vis_maps(self,car_trajectory, refined_global_map, pred_save_path=None):
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
        slb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, refined_global_map)
    
    def merge_by_frame(self,global_map, proximity_th, pre =True, denoise = True):
        sub_map = [[],[],[]]
        start = 5
        # bar = Bar('merge', max=(len(global_map[0]) - start - 1))
        # start_time = time.time()
        for cate in range(3):
            sub_map[cate] += (global_map[cate][0:start])

        for i in range(start,len(global_map[0]) - 1):
            for cate in range(3):
                sub_map[cate].append(global_map[cate][i + 1])
            sub_map = slb.refine_token_v3(sub_map, proximity_th, pre =pre, denoise = denoise)
            # bar.next()
            # elapsed_time = time.time() - start_time
            # bar.message = f'merge: {elapsed_time:.2f} s'
        # bar.finish()

        return sub_map

    def global_map_stitch(self,global_data, if_by_frame = False, if_vis=False, pred_save_path=None):
        global_map, car_trajectory = self.preprocess_data(global_data)
        # global_map[0] = []
        # global_map[2] = []

        if_by_frame = False
        if if_by_frame:
            refined_global_map = self.merge_by_frame(global_map, proximity_th = 0.8, pre =False, denoise = False)
        else:
            refined_global_map = slb.refine_token_v2(global_map, proximity_th = 0.8, pre =True, denoise = True)
            
        merge_result = self.format_result(refined_global_map)
        self.stitch_data = merge_result

        if if_vis:
            self.vis_maps(car_trajectory, refined_global_map, pred_save_path)
            
        return merge_result