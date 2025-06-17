import sys

from ..map_stitch_libs import global_lib as glb
from ..map_stitch_libs  import Stitch_lib as slb

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev
from progress.bar import Bar as bar
import time
from shapely.geometry import LineString

class GlobalMapNet_Global:
    def __init__(self):
        self.id2cat=['lane','ped', 'road']
        self.map_builder = glb.MapBuilder()
        self.stitch_data = None
        self.global_map_config = glb.global_map_config
    def pre_process(self,global_data):
        car_trajectory = []
        local_map_elements = []
        for frame in global_data:
            local_map = {}
            loc = list(frame['gt']['pose'])
            yaw = frame['gt']['yaw'] - np.pi / 2
            vectors = []

            for i in range(len(frame['pred']['label'])):
                vec = {}
                # if frame['pred']['label'][i] == 2:
                #     continue
                vec['category'] = self.id2cat[frame['pred']['label'][i]]
                vec['coords'] = frame['pred']['box'][i]
                vectors.append(vec)
            local_map['meta'] = vectors
            # import ipdb; ipdb.set_trace()

            local_map['pose'] = loc + [0] + [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
            local_map_elements.append(local_map)

        return local_map_elements, car_trajectory
    
    def format_result(self, global_map_elements):
        pred_datas = {'divider': [], 'ped_crossing': [], 'boundary': []}
        for pred_element in global_map_elements:
            label = pred_element['category']
            if label == 'ped': # ped_crossing
                key_label = 'ped_crossing'
            elif label == 'lane': # divider
                key_label = 'divider'
            elif label == 'road': # boundary
                key_label = 'boundary'
        
            # get the vectors belongs to the same instance
            polyline = LineString(pred_element['coords'] )
            simplify = 0.5
            polyline = np.array(polyline.simplify(simplify).coords)
            pred_datas[key_label].append(polyline)
        return pred_datas
    
    def vis_maps(car_trajectory, global_map_elements, pred_save_path=None):
        all_points = []
        for vecs in global_map_elements:
            points = vecs['coords']
            all_points.append(points)
        all_points = np.concatenate(all_points, axis=0)
        x_min = all_points[:,0].min()
        x_max = all_points[:,0].max()
        y_min = all_points[:,1].min()
        y_max = all_points[:,1].max()
        glb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, global_map_elements)


    def global_map_stitch(self, global_data, if_vis=False, pred_save_path=None):
        map_name = 'global_test' 
        self.map_builder.init_global_map(map_name)

        local_map_elements, car_trajectory = self.pre_process(global_data)
        for local_map in local_map_elements:
            self.map_builder.update_global_map( map_name, local_map['meta'], local_map['pose'], from_ego_coords=True, \
                                replace_mode=glb.MapReplaceMode(self.global_map_config['replace_mode']), \
                                nms_purge_mode=glb.MapNMSPurgeMode(self.global_map_config['nms_purge_mode']),\
                                nms_score_mode=glb.MapNMSScoreMode(self.global_map_config['nms_score_mode']), \
                                **self.global_map_config['update_kwargs'])
    
        global_map_elements = self.map_builder.global_maps[map_name]['map_elements']
        merge_result = self.format_result(global_map_elements)
        self.stitch_data = merge_result

        if if_vis:  
            self.vis_maps(car_trajectory, global_map_elements, pred_save_path)
        
        return merge_result