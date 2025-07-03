# import CustomMetrics as CM    
import sys
sys.path.append('..')
import evaluate.CustomMetrics as CM
import evaluate.visual as visual

from globalmapnet_global.bound_divider import globalmapnet_global as GLOBALMAPNET_GLOBAL
from maptracker_global.bound_divider import maptracker_global as MAPTRACKER_GLOBAL
from polymerge_global.bound_divider import polymerge_global as POLYMERGE_GLOBAL
from ours_global.bound_divider import ours_global as OURS_GLOBAL

import pickle as pkl
from shapely.geometry import LineString
import numpy as np
import os
import time
from progress.bar import Bar
from shapely.geometry import LineString
import matplotlib.pyplot as plt


categories = ['divider', 'ped_crossing', 'boundary']


def get_method_dict(method_list):
    method_dict = {}
    if 'globalmapnet' in method_list:
        method_dict['globalmapnet'] = GLOBALMAPNET_GLOBAL.GlobalMapNet_Global()

    if'maptracker' in method_list:
        method_dict['maptracker'] = MAPTRACKER_GLOBAL.MapTracker_Global()

    if'polymerge' in method_list:
        method_dict['polymerge'] = POLYMERGE_GLOBAL.PolyMerge_Global()

    if'ours' in method_list:
        method_dict['ours'] = OURS_GLOBAL.Ours_Global()

    return method_dict

def fromat_gt_scene(gt_scene_data):
    gt_datas = {'divider': [], 'ped_crossing': [], 'boundary': []}
    for gt_element_by_cate in gt_scene_data:
        label = gt_element_by_cate['category']
        if label not in gt_datas:
            continue
        # import ipdb; ipdb.set_trace()
        if gt_element_by_cate['instances'] == None:
            continue
        
        for global_coords in gt_element_by_cate['instances']:
            if len(global_coords) == 0:
                # import ipdb; ipdb.set_trace()
                continue
            # print(global_coords)
            polyline = LineString(global_coords)
            simplify = 0.5
            polyline = np.array(polyline.simplify(simplify).coords)

            gt_datas[label].append(polyline)
    # import ipdb; ipdb.set_trace()
    return gt_datas

def draw_unmerge(dataset_name):
    all_pred_data_path = '/data/result_keep_all_' + dataset_name +'_glb.pkl'
    with open(all_pred_data_path, 'rb') as f:
        all_pred_data = pkl.load(f)
    
    method = OURS_GLOBAL.Ours_Global()

    simplify = 0.5
    line_opacity = 0.75

    all_points = []

    store_gt = 'save_' + dataset_name +'/unmerge'
    if not os.path.exists(store_gt):
        # 如果文件夹不存在，则创建
        os.makedirs(store_gt)


    for scene_name, scene_info in all_pred_data.items():

        local_data = scene_info['local'][::3]
        # import ipdb; ipdb.set_trace()

        pred_save_path = store_gt + '/vis_result_unmerge_' + scene_name + '.png'

        global_data, car_traj = method.preprocess_data(local_data)
        # import ipdb; ipdb.set_trace()
        method.vis_maps( car_traj, global_data, pred_save_path=pred_save_path)

        # print("image saved to : ", pred_save_path)

def format_all_global_datas(dataset_name, method_list):
    all_pred_data_path = '/data/result_keep_all_' + dataset_name +'_glb.pkl'
    with open(all_pred_data_path, 'rb') as f:
        all_pred_data = pkl.load(f)
    
    method_dict = get_method_dict(method_list)

    # all_pred_data = dict(list(all_pred_data.items())[32:33])

    format_eval_all_dict = {}
    start_time = time.time()
    
    for method_name in method_list:
        method = method_dict[method_name]

        format_eval_all_dict_per_method = {}
        bar = Bar(method_name, max=len(all_pred_data))
        for scene_name, scene_info in all_pred_data.items():

            format_eval_scene_dict = {}
            local_data = scene_info['local'][::3]
            global_data = scene_info['global']
            # import ipdb; ipdb.set_trace()

            merged_scenes = method.global_map_stitch(local_data)
            scene_gt_data = fromat_gt_scene(global_data)

            format_eval_scene_dict['pred'] = merged_scenes
            format_eval_scene_dict['gt'] = scene_gt_data

            format_eval_all_dict_per_method[scene_name] = format_eval_scene_dict

            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f"{method_name}: {elapsed_time:.2f} s"
        bar.finish()
        format_eval_all_dict[method_name] = format_eval_all_dict_per_method

    return format_eval_all_dict

def eval_all_methods(dataset_name, format_eval_all_dict, method_list, metric):

    final_eval_results = {}
    save_global = True

    for method_name in method_list:
        format_eval_dict = format_eval_all_dict[method_name]
        visual.vis_per_method(dataset_name, format_eval_dict, method_name)
        if save_global:
            # visual.vis_gt(dataset_name, format_eval_dict, method_name)
            save_global = False
        eval_results_per_method = CM.evaluate_per_method(format_eval_dict, method_name, metric)
        final_eval_results[method_name] = eval_results_per_method
    
    return final_eval_results

if __name__ == '__main__':

    # method_list = ['globalmapnet','maptracker', 'polymerge', 'ours']
    method_list = ['polymerge']
    method_list = ['polymerge', 'ours']
    method_list = ['maptracker', 'polymerge', 'ours', 'globalmapnet']
    method_list = ['ours']
    # method_list = ['polymerge']
    # method_list = ['polymerge', 'ours', 'globalmapnet','maptracker']
    # method_list = ['maptracker']

    dataset_name = 'nuscenes'
    # dataset_name = 'weijing'
    # dataset_name = 'argo'

    metric='frechet'
    metric='chamfer'


    merged_data_path = dataset_name + '_merged_data' + '_'.join(method_list) + '.pkl'
    eval_result_path = dataset_name + '_eval_result' + '_'.join(method_list) + '.pkl'

    # draw_unmerge(dataset_name)S
    # import ipdb; ipdb.set_trace()


    if os.path.exists( merged_data_path) and os.path.isfile( merged_data_path):
        with open( merged_data_path, 'rb') as f:
            format_datas = pkl.load(f)
    else:
        format_datas = format_all_global_datas(dataset_name, method_list)
        with open( merged_data_path, 'wb') as f:
            pkl.dump(format_datas, f)

    final_eval_results = eval_all_methods(dataset_name, format_datas, method_list, metric)
    with open( eval_result_path, 'wb') as f:
            pkl.dump(final_eval_results, f)

