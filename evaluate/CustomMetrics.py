
from functools import partial
import numpy as np
from multiprocessing import Pool

from .AP import global_match, average_precision
import prettytable
import json
import time
from functools import cached_property
from shapely.geometry import LineString
from shapely.ops import unary_union
from numpy.typing import NDArray
from typing import Dict, List, Optional
from logging import Logger
from copy import deepcopy
import os
from IPython import embed
import pickle as pkl
from shapely.geometry import LineString 
from progress.bar import Bar as bar


categories = ['divider', 'ped_crossing', 'boundary']
THRESHOLDS_CHAMFER = [0.5, 1.0, 1.5]
# THRESHOLDS_FRECHET = [1.0, 2.0, 3.0]
THRESHOLDS_FRECHET = [5.0, 10.0, 15.0]

INTERP_NUM = 200
# INTERP_NUM = 400

def _evaluate_single(   pred_vectors: List, 
                         groundtruth: List, 
                         thresholds: List, 
                         metric: str='metric') -> Dict[int, NDArray]:
        ''' Do single-frame matching for one class.
        
        Args:
            pred_vectors (List): List[vector(ndarray) (different length)], 
            scores (List): List[score(float)]
            groundtruth (List): List of vectors
            thresholds (List): List of thresholds
        
        Returns:
            tp_fp_score_by_thr (Dict): matching results at different thresholds
                e.g. {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        '''

        pred_lines = []

        # interpolate predictions
        for vector in pred_vectors:
            vector = np.array(vector)
            vector_interp = interp_fixed_num(vector, INTERP_NUM)
            pred_lines.append(vector_interp)
        if pred_lines:
            pred_lines = np.stack(pred_lines)
        else:
            pred_lines = np.zeros((0, INTERP_NUM, 2))

        # interpolate groundtruth
        gt_lines = []
        for vector in groundtruth:
            vector_interp = interp_fixed_num(vector, INTERP_NUM)
            gt_lines.append(vector_interp)
        if gt_lines:
            gt_lines = np.stack(gt_lines)
        else:
            gt_lines = np.zeros((0, INTERP_NUM, 2))
        
        #match这里需要重新设计
        tp_fp_list = global_match(pred_lines[..., :2], gt_lines[..., :2], thresholds, metric) # (M, 2)

        tp_fp_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tp_fp_list[i]
            tp_fp = np.hstack([tp[:, None], fp[:, None]])
            tp_fp_by_thr[thr] = tp_fp
        
        
        return tp_fp_by_thr # {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}

def interp_fixed_num(vector: NDArray, 
                         num_pts: int) -> NDArray:
        ''' Interpolate a polyline.
        
        Args:
            vector (array): line coordinates, shape (M, 2)
            num_pts (int): 
        
        Returns:
            sampled_points (array): interpolated coordinates
        '''
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_pts)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()
        
        return sampled_points

def evaluate_per_method(eval_data_per_method, method_name, eval_name = 'frechet'):

    if eval_name == 'frechet':
        thresholds=THRESHOLDS_FRECHET
        metric ='frechet'
    # thresholds=THRESHOLDS_FRECHET
    else:
        metric ='chamfer'
        thresholds=THRESHOLDS_CHAMFER
    
    pbar = bar(method_name + ' eval', max=len(eval_data_per_method))
    eval_results = {}
    start_time = time.time()

    # all_eval_data = dict(list(all_eval_data.items())[:5])
    for scene_name, eval_data in eval_data_per_method.items():

        pred_global_maps = eval_data['pred']
        gt_global_maps = eval_data['gt']

        result_dicts = {}

        for label in categories:
            pred_map_vectors = pred_global_maps[label]
            gt_map_vectors = gt_global_maps[label]

            sample = (pred_map_vectors, gt_map_vectors)
            fn = partial(_evaluate_single, thresholds=thresholds, metric=metric)
            tpfp = fn(*sample) 

            tps = {thr: 0 for thr in thresholds}
            fps = {thr: 0 for thr in thresholds}
            fns = {thr: 0 for thr in thresholds}

            result_dict = {} 

            for thr in thresholds:
                tp_fp = tpfp[thr]
                column_sum = np.sum(tp_fp, axis=0)

                tp = column_sum[0]
                fp = column_sum[1]
                Fn = len(gt_map_vectors) - tp


                tps[thr] = tp
                fps[thr] = fp
                fns[thr] = Fn

            result_dict['tps'] = tps
            result_dict['fps'] = fps
            result_dict['fns'] = fns
            result_dict['gt_nums'] = len(gt_map_vectors)
            result_dict['pred_nums'] = len(pred_map_vectors)
            result_dicts[label] = result_dict

        eval_results[scene_name] = result_dicts

        elapsed_time = time.time() - start_time
        pbar.message = f"{method_name} eval: {elapsed_time:.2f} s"
        pbar.next()

    pbar.finish()

    print('formatting ',method_name,' eval results...')

    all_tp = {category: {thr: 0 for thr in thresholds} for category in categories}
    all_fp = {category: {thr: 0 for thr in thresholds} for category in categories}
    all_fn = {category: {thr: 0 for thr in thresholds} for category in categories}
    F1_scores = {category: {thr: 0 for thr in thresholds} for category in categories}

    all_gt_number = {category: 0 for category in categories}
    all_pred_number = {category: 0 for category in categories}

    for scene_name, eval_result in eval_results.items():
        for label in categories:
            result_dict = eval_result[label]
            for thr in thresholds:
                all_tp[label][thr] += result_dict['tps'][thr]
                all_fp[label][thr] += result_dict['fps'][thr]
                all_fn[label][thr] += result_dict['fns'][thr]

            all_gt_number[label] += result_dict['gt_nums']
            all_pred_number[label] += result_dict['pred_nums']

    for label in categories:
        for thr in thresholds:
            tp = all_tp[label][thr]
            fp = all_fp[label][thr]
            fn = all_fn[label][thr]

            eps = np.finfo(np.float32).eps
            recall = tp / np.maximum(fn + tp, eps)
            precision = tp / np.maximum((tp + fp), eps)

            F1_score = 2 * precision * recall / (precision + recall + eps)

            F1_scores[label][thr] = F1_score

    save_eval_result = {
        'eval_results': eval_results,
        'F1_scores': F1_scores,
        'all_gt_number': all_gt_number,
        'all_pred_number': all_pred_number,
    }

    # print results
    table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
            [f'F1_scores@{thr}' for thr in thresholds] + ['F1_scores'])
    for label in categories:
        table.add_row([
            label, 
            all_pred_number[label],
            all_gt_number[label],
            *[round(F1_scores[label][thr], 4) for thr in thresholds],
            round(np.mean(list(F1_scores[label].values())), 4),
        ])
    print(table)
     
    return save_eval_result


def evaluate(all_eval_data, thresholds=THRESHOLDS_CHAMFER, metric: str='chamfer'):
    
    pbar = bar('eval', max=len(all_eval_data))
    eval_results = {}
    start_time = time.time()

    # all_eval_data = dict(list(all_eval_data.items())[:5])
    for scene_name, eval_data in all_eval_data.items():

        pred_global_maps = eval_data['pred']
        gt_global_maps = eval_data['gt']

        result_dicts = {}

        for label in categories:
            pred_map_vectors = pred_global_maps[label]
            gt_map_vectors = gt_global_maps[label]
            
            #整合传参
            sample = (pred_map_vectors, gt_map_vectors)
            

            fn = partial(_evaluate_single, thresholds=thresholds, metric=metric)
            # import ipdb;ipdb.set_trace()
            tpfp = fn(*sample) 

            # recalls = {}
            # precisions = {}
            # F1_scores = {} 

            tps = {thr: 0 for thr in thresholds}
            fps = {thr: 0 for thr in thresholds}
            fns = {thr: 0 for thr in thresholds}

            result_dict = {} 

            for thr in thresholds:
                tp_fp = tpfp[thr]
                column_sum = np.sum(tp_fp, axis=0)

                tp = column_sum[0]
                fp = column_sum[1]

                Fn = len(gt_map_vectors) - tp

                

                # eps = np.finfo(np.float32).eps
                # recall = tp / np.maximum(Fn + tp, eps)
                # precision = tp / np.maximum((tp + fp), eps)

                # F1_score = 2 * precision * recall / (precision + recall + eps)

                # recalls[thr] = recall
                # precisions[thr] = precision
                # F1_scores[thr] = F1_score

                tps[thr] = tp
                fps[thr] = fp
                fns[thr] = Fn

            # result_dict['recalls'] = recalls
            # result_dict['precisions'] = precisions
            # result_dict['F1_scores'] = F1_scores

            result_dict['tps'] = tps
            result_dict['fps'] = fps
            result_dict['fns'] = fns
            result_dict['gt_nums'] = len(gt_map_vectors)
            result_dict['pred_nums'] = len(pred_map_vectors)
            result_dicts[label] = result_dict

        eval_results[scene_name] = result_dicts

        elapsed_time = time.time() - start_time
        pbar.message = f'eval: {elapsed_time:.2f} s'
        pbar.next()

    pbar.finish()



    all_tp = {category: {thr: 0 for thr in thresholds} for category in categories}
    all_fp = {category: {thr: 0 for thr in thresholds} for category in categories}
    all_fn = {category: {thr: 0 for thr in thresholds} for category in categories}
    F1_scores = {category: {thr: 0 for thr in thresholds} for category in categories}

    all_gt_number = {category: 0 for category in categories}
    all_pred_number = {category: 0 for category in categories}

    for scene_name, eval_result in eval_results.items():
        for label in categories:
            result_dict = eval_result[label]
            for thr in thresholds:
                all_tp[label][thr] += result_dict['tps'][thr]
                all_fp[label][thr] += result_dict['fps'][thr]
                all_fn[label][thr] += result_dict['fns'][thr]
            
            # import ipdb;ipdb.set_trace()


            all_gt_number[label] += result_dict['gt_nums']
            all_pred_number[label] += result_dict['pred_nums']

    for label in categories:
        for thr in thresholds:
            tp = all_tp[label][thr]
            fp = all_fp[label][thr]
            fn = all_fn[label][thr]

            eps = np.finfo(np.float32).eps
            recall = tp / np.maximum(fn + tp, eps)
            precision = tp / np.maximum((tp + fp), eps)

            F1_score = 2 * precision * recall / (precision + recall + eps)

            F1_scores[label][thr] = F1_score
    # import ipdb;ipdb.set_trace()



    save_eval_result = {
        'eval_results': eval_results,
        'F1_scores': F1_scores,
        'all_gt_number': all_gt_number,
        'all_pred_number': all_pred_number,
    }

    # print results
    table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
            [f'F1_scores@{thr}' for thr in thresholds] + ['F1_scores'])
    for label in categories:
        table.add_row([
            label, 
            all_pred_number[label],
            all_gt_number[label],
            *[round(F1_scores[label][thr], 4) for thr in thresholds],
            round(np.mean(list(F1_scores[label].values())), 4),
        ])
    print(table)
     
    return save_eval_result

if __name__ == '__main__':

    # eval_path = 'eval_globalmapnet_scene-0077.pkl'
    eval_path = 'nuscenes_eval_globalmapnet.pkl'
    save_results_path = 'result_eval_globalmapnet_nuscenes.pkl'

    with open(eval_path, 'rb') as f:
        eval_data = pkl.load(f)

    eval_results = evaluate(eval_data, THRESHOLDS_CHAMFER, metric='chamfer')

    # import ipdb;ipdb.set_trace()


    with open(save_results_path, 'wb') as f:
        pkl.dump(eval_results, f)


