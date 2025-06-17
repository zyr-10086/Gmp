import sys
sys.path.append('..')
from map_stitch_libs import global_lib as glb
from map_stitch_libs  import Stitch_lib as slb

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev
from progress.bar import Bar as bar
import time


if __name__ == '__main__':

    pkl_name = 's1'

    id2cat=['lane','ped', 'road']

    roi_size = (60,30)
    global_map_config = dict(
        id2cat=['ped', 'lane', 'road'],
        score_threshold=0.4,
        update_interval_test=1,
        update_indices_in_batch_test=None,
        patch_size=roi_size,
        root_dir=r'./cache/global_maps',
        threshold=0.05,
        map_name='streammapnet',
        adjust_rot_angle=None,
        replace_mode=1, # plugin/models/globalmapnet/map_utils/map_builder.py
        nms_purge_mode=2, # plugin/models/globalmapnet/map_utils/functional/map_nms.py
        nms_score_mode=1, # ...
        update_kwargs={
            'threshold': [
                0.05,
                {
                    'road': 2.0,
                    'lane': 1.0,
                    'ped': 0.5
                },
                1.0],
            'sample_num': 100,
            'simplify': True,
            'buffer_distance': 
            {
                'road': 2.0,
                'lane': 1.0,
                'ped': 0.5
            },
            'biou_threshold': 0.1}
        )

    # data_path = '../../../local_data/nuscenes/result_keep_scene-0077.pkl'
    # data_path = '../../../local_data/datas_v2/result_keep_s1.pkl'
    data_path = '../../../local_data/nuscenes/result_keep_all_nuscenes.pkl'
    data = slb.load_data(data_path)
    # data = data[::2]

    all_eval_data = {}
    # data = dict(list(data.items())[:5])

    pbar = bar('pred', max=len(data))
    start_time = time.time()

    for scene_name, scene_info in data.items():
        car_trajectory = []
        local_map_elements = []
        for frame in scene_info['local']:
            local_map = {}
            loc = frame['gt']['pose']
            yaw = frame['gt']['yaw'] - np.pi / 2
            vectors = []

            for i in range(len(frame['pred']['label'])):
                vec = {}
                # if frame['pred']['label'][i] == 2:
                #     continue
                vec['category'] = id2cat[frame['pred']['label'][i]]
                vec['coords'] = frame['pred']['box'][i]
                vectors.append(vec)
            local_map['meta'] = vectors

            local_map['pose'] = loc + [0] + [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            car_trajectory.append([np.array(loc), yaw * 180 / np.pi])



            local_map_elements.append(local_map)

        map_name = 'global_test'
        map_builder = glb.MapBuilder()
        map_builder.init_global_map(map_name)

        for local_map in local_map_elements:
            map_builder.update_global_map( map_name, local_map['meta'], local_map['pose'], from_ego_coords=True, \
                                replace_mode=glb.MapReplaceMode(global_map_config['replace_mode']), \
                                nms_purge_mode=glb.MapNMSPurgeMode(global_map_config['nms_purge_mode']),\
                                nms_score_mode=glb.MapNMSScoreMode(global_map_config['nms_score_mode']), \
                                **global_map_config['update_kwargs'])
    
        global_map_elements = map_builder.global_maps[map_name]['map_elements']
        # pred_save_path = 'merge_global_result_' + pkl_name + '.png'

        # all_points = []
        # for vecs in global_map_elements:
        #     points = vecs['coords']
        #     all_points.append(points)
        # all_points = np.concatenate(all_points, axis=0)
        # x_min = all_points[:,0].min()
        # x_max = all_points[:,0].max()
        # y_min = all_points[:,1].min()
        # y_max = all_points[:,1].max()

        # eval_export_path = 'eval_globalmapnet_scene-0077.pkl'
        all_eval_data[scene_name] = (global_map_elements, scene_info['global'])
        # glb.export_eval_data(global_map_elements, data['global'], export_path=eval_export_path)
        # glb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, global_map_elements)
        elapsed_time = time.time() - start_time
        pbar.message = f'eval: {elapsed_time:.2f} s'
        pbar.next()
    pbar.finish()
    
    eval_export_path = 'nuscenes_eval_globalmapnet.pkl'
    glb.export_eval_data_all(all_eval_data,  export_path=eval_export_path)
    
   
    # import ipdb; ipdb.set_trace()
