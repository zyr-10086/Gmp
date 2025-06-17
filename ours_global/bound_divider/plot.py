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
from shapely.geometry import LineString, Point

def preprocess_glb_map( instances, threshold=1.0):
    filter = []
    line_objects = [LineString(polyline) for polyline in instances]

    while line_objects:
        line = line_objects.pop(0)  # 获取第一个折线
        merged = False

        # 尝试与其他线段合并
        for i, other_line in enumerate(line_objects[:]):
            # 获取两条折线的端点
            start_point = line.coords[0]
            end_point = line.coords[-1]
            other_start_point = other_line.coords[0]
            other_end_point = other_line.coords[-1]

            # 计算端点之间的距离
            distances = [
                Point(start_point).distance(Point(other_end_point)),  # 当前线的起点和其他线的终点
                Point(end_point).distance(Point(other_start_point)),  # 当前线的终点和其他线的起点
            ]

            # 判断是否小于阈值
            if min(distances) < threshold:
                print()
                # 合并线段：按顺序将线段连接起来
                if distances[0] > distances[1]:
                    merged_line = LineString(list(line.coords) + list(other_line.coords)[1:])
                else:
                    merged_line = LineString(list(other_line.coords) + list(line.coords)[1:])
                
                # 更新合并后的线段
                line_objects[i] = merged_line
                merged = True
                break  # 进行合并后退出当前循环，重新开始判断

        # 如果没有合并，则将当前线段添加到结果中
        if not merged:
            filter.append(np.array(line.coords))
    return filter

if __name__ == '__main__':

    data_path = '../../../local_data/nuscenes/result_keep_all_nuscenes.pkl'
    data_path = '/home/zyr/MapTR/result_keep_all_weijing_glb.pkl'
    save = 0


    data = slb.load_data(data_path)


    for scene_name , scene_data in data.items():
        global_map = [[], [], []]
        car_trajectory = []
        # # import ipdb;ipdb.set_trace()
        # for index in range(0,len(data[scene_name]['local'])):
        #     frame = data[scene_name]['local'][index]
        #     loc = frame['gt']['pose']
        #     yaw = frame['gt']['yaw'] - np.pi / 2
        #     car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
        #     message = slb.preprocess_frame(frame)        
        #     slb.update_global_map(global_map, message, type='gt')

        for i in range(3):
            frame = []
            glb_data = preprocess_glb_map(scene_data['global'][i]['instances'])
            glb_data = scene_data['global'][i]['instances']
            for ins in glb_data:
                # ins_line = LineString(ins)
                # if ins_line.length < 1:
                #     continue
                frame.append(ins)
            global_map[i].append(frame)

        import ipdb;ipdb.set_trace()
        refined_global_map = global_map

        # for i in range(0, len(refined_global_map)):
        #     refined_global_map[i].append(global_map_2[i][0])

        all_points = []

        for catogory in refined_global_map:
            for frame in catogory:
                for vec in frame:
                    all_points.append(vec)
        # import ipdb;ipdb.set_trace()

        all_points = np.concatenate(all_points, axis=0)
        x_min = all_points[:,0].min()
        x_max = all_points[:,0].max()
        y_min = all_points[:,1].min()
        y_max = all_points[:,1].max()


        pred_save_path = 'gts/gt_'+ scene_name + '.png'
        slb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, refined_global_map)