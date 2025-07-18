# scene-level global map evaluation
# adapted from plugin/datasets/evaluation/vector_eval.py
from functools import partial
import numpy as np
from multiprocessing import Pool
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from .AP import instance_match, average_precision
import prettytable
import json
from time import time
from functools import cached_property
from shapely.geometry import LineString
from shapely.ops import unary_union
from numpy.typing import NDArray
from typing import Dict, List, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy
import os
from IPython import embed

from .ego import generate_patch_box


INTERP_NUM = 200 # number of points to interpolate during evaluation
THRESHOLDS = [0.5, 1.0, 1.5] # AP thresholds
N_WORKERS = 16 # num workers to parallel
SAMPLE_DIST = 0.15

class SceneEvaluate(object):
    """Evaluator for scene-level global map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, root_dir, id2cat, dataset_cfg: Config, n_workers: int=N_WORKERS) -> None:
        self.root_dir = root_dir
        self.id2cat = {idx: cat for idx, cat in enumerate(id2cat)}
        self.cat2id = {v: k for k, v in self.id2cat.items()}

        self.dataset = build_dataset(dataset_cfg)
        self.n_workers = n_workers
        self.new_split = 'newsplit' in self.dataset.ann_file
        self.roi_size = self.dataset.roi_size
        if self.roi_size == (60, 30):
            self.thresholds = [0.5, 1.0, 1.5]
        elif self.roi_size == (100, 50):
            self.thresholds = [1.0, 1.5, 2.0]
        self.location_dict = {'av2': [], 'nusc': [
            'boston-seaport',
            'singapore-hollandvillage',
            'singapore-onenorth',
            'singapore-queenstown']}
        if 'av2' in self.dataset.ann_file:
            self.dataset_name = 'av2'
        else:
            self.dataset_name = 'nusc'
        
    @cached_property
    def gts(self) -> Dict[str, Dict[int, List[NDArray]]]:
        roi_size = self.dataset.roi_size

        gts = dict()
        if self.dataset_name == 'nusc':
            for location in self.location_dict[self.dataset_name]:
                with open(os.path.join(self.root_dir, f'{location}_gt.json'), 'r') as f:
                    global_map = json.load(f)
                gts[location] = global_map
        else:
            # av2, use scene token as key
            with open(os.path.join(self.root_dir, f'av2_all_gt.json'), 'r') as f:
                gts = json.load(f)

        return gts
    
    def interp_fixed_num(self, 
                         vector: NDArray, 
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
    
    def interp_fixed_dist(self, 
                          vector: NDArray,
                          sample_dist: float) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            vector (LineString): vector
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''
        line = LineString(vector)
        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

    def _evaluate_single(self, 
                         pred_vectors: List, 
                         scores: List, 
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
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            pred_lines.append(vector_interp)
        if pred_lines:
            pred_lines = np.stack(pred_lines)
        else:
            pred_lines = np.zeros((0, INTERP_NUM, 2))

        # interpolate groundtruth
        gt_lines = []
        for vector in groundtruth:
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            gt_lines.append(vector_interp)
        if gt_lines:
            gt_lines = np.stack(gt_lines)
        else:
            gt_lines = np.zeros((0, INTERP_NUM, 2))
        
        scores = np.array(scores)
        tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds, metric) # (M, 2)
        tp_fp_score_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tp_fp_list[i]
            tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score
        
        return tp_fp_score_by_thr # {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        
    def evaluate(self, 
                 global_maps, 
                 metric: str='chamfer', 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. We use multi-worker to speed up.
        
        Args:
            global_maps (dict): a dict of predicted global maps
            metric (str): distance metric. Default: 'chamfer'
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            new_result_dict (Dict): evaluation results. AP by categories.
        '''

        # re-group samples and gt by label
        samples_by_cls = {label: [] for label in self.id2cat.keys()}
        num_gts = {label: 0 for label in self.id2cat.keys()}
        num_preds = {label: 0 for label in self.id2cat.keys()}
        
        for map_name in global_maps.keys():
            global_map = global_maps[map_name]
            if self.dataset_name == 'nusc':
                location = global_map['meta_info']['location']
                gt = self.gts[location]
            else:
                # av2, use scene name as key
                scene_name = global_map['meta_info']['scene_name']
                gt = self.gts[scene_name]
            
            # for every sample
            vectors_by_cls = {label: [] for label in self.id2cat.keys()}
            scores_by_cls = {label: [] for label in self.id2cat.keys()}

            for idx in range(len(global_map['map_elements'])):
                # i-th pred line in sample
                map_element = global_map['map_elements'][idx]
                label = self.cat2id[map_element['category']]
                vector = map_element['coords']
                score = map_element['details']['score']

                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)
            
            # cut out gt scene map by pose and roi
            poses = global_map['poses']
            patch_boxes = [generate_patch_box(self.roi_size, pose) for pose in poses]
            scene_region = unary_union(patch_boxes)

            gt_by_cls = {label: [] for label in self.id2cat.keys()}
            gt_map_elements = list()
            for gt_map_element in gt['map_elements']:
                label = self.cat2id[gt_map_element['category']]
                gt_coords = gt_map_element['coords']
                gt_line = LineString(gt_coords)
                if scene_region != None:
                    gt_line = scene_region.intersection(gt_line)
                if not gt_line.is_empty:
                    if gt_line.geom_type == 'MultiLineString':
                        for single_polyline in gt_line.geoms:
                            gt_by_cls[label].append(list(single_polyline.coords))
                    else:
                        gt_by_cls[label].append(list(gt_line.coords))

            for label in self.id2cat.keys():
                new_sample = (vectors_by_cls[label], scores_by_cls[label], gt_by_cls[label])
                num_gts[label] += len(gt_by_cls[label])
                num_preds[label] += len(scores_by_cls[label])
                samples_by_cls[label].append(new_sample)
        
        # no change below

        result_dict = {}

        print(f'\nevaluating {len(self.id2cat)} categories...')
        start = time()
        if self.n_workers > 0:
            pool = Pool(self.n_workers)
        
        sum_mAP = 0
        pbar = mmcv.ProgressBar(len(self.id2cat))
        for label in self.id2cat.keys():
            samples = samples_by_cls[label] # List[(pred_lines, scores, gts)]
            result_dict[self.id2cat[label]] = {
                'num_gts': num_gts[label],
                'num_preds': num_preds[label]
            }
            sum_AP = 0

            fn = partial(self._evaluate_single, thresholds=self.thresholds, metric=metric)
            if self.n_workers > 0:
                tpfp_score_list = pool.starmap(fn, samples)
            else:
                tpfp_score_list = []
                for sample in samples:
                    tpfp_score_list.append(fn(*sample))
            # pbar = mmcv.ProgressBar(len(samples))
            # tpfp_score_list = []
            # for sample in samples:
            #     tpfp_score_list.append(fn(*sample))
                # pbar.update()
            
            for thr in self.thresholds:
                tp_fp_score = [i[thr] for i in tpfp_score_list]
                tp_fp_score = np.vstack(tp_fp_score) # (num_dets, 3)
                sort_inds = np.argsort(-tp_fp_score[:, -1])

                tp = tp_fp_score[sort_inds, 0] # (num_dets,)
                fp = tp_fp_score[sort_inds, 1] # (num_dets,)
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts[label], eps)
                precisions = tp / np.maximum((tp + fp), eps)

                AP = average_precision(recalls, precisions, 'area')
                sum_AP += AP
                result_dict[self.id2cat[label]].update({f'AP@{thr}': AP})

            pbar.update()
            
            AP = sum_AP / len(self.thresholds)
            sum_mAP += AP

            result_dict[self.id2cat[label]].update({f'AP': AP})
        
        if self.n_workers > 0:
            pool.close()
        
        mAP = sum_mAP / len(self.id2cat.keys())
        result_dict.update({'mAP': mAP})
        
        print(f"finished in {time() - start:.2f}s")

        # print results
        table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
                [f'AP@{thr}' for thr in self.thresholds] + ['AP'])
        for label in self.id2cat.keys():
            table.add_row([
                self.id2cat[label], 
                result_dict[self.id2cat[label]]['num_preds'],
                result_dict[self.id2cat[label]]['num_gts'],
                *[round(result_dict[self.id2cat[label]][f'AP@{thr}'], 4) for thr in self.thresholds],
                round(result_dict[self.id2cat[label]]['AP'], 4),
            ])
        
        from mmcv.utils import print_log
        print_log('\n'+str(table), logger=logger)
        mAP_normal = 0
        for label in self.id2cat.keys():
            for thr in self.thresholds:
                mAP_normal += result_dict[self.id2cat[label]][f'AP@{thr}']
        mAP_normal = mAP_normal / 9

        print_log(f'mAP_normal = {mAP_normal:.4f}\n', logger=logger)
        # print_log(f'mAP_hard = {mAP_easy:.4f}\n', logger=logger)

        new_result_dict = {}
        for name in self.cat2id:
            new_result_dict[name] = result_dict[name]['AP']

        return new_result_dict