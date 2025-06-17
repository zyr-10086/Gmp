import sys
import os

import argparse     

import matplotlib.transforms as transforms

import cv2
import torch
import numpy as np
from PIL import Image
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull
from PIL import Image
import cv2
import imageio
import math
from .utils import *
from .match_utils import *

def combine_images_with_labels(image_paths, labels, output_path, font_scale=0.5, font_color=(0, 0, 0)):
    # Load images
    images = [cv2.imread(path) for path in image_paths]
    
    # Determine the maximum dimensions
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    
    # Create a blank white canvas to hold the 2x2 grid of images
    final_image = np.ones((max_height * 1, max_width * 2, 3), dtype=np.uint8) * 255
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, img in enumerate(images):
        # Resize image if necessary
        img = cv2.resize(img, (max_width, max_height))
        
        # Calculate position for each image
        x_offset = (i % 2) * max_width
        y_offset = (i // 2) * max_height
        
        # Place image in the canvas
        final_image[y_offset:y_offset+max_height, x_offset:x_offset+max_width] = img
        
        # Add label
        cv2.putText(final_image, labels[i], (x_offset + 5, y_offset + 15), font, font_scale, font_color, 1, cv2.LINE_AA)
    
    # Save the final image
    cv2.imwrite(output_path, final_image)


def merge_corssing(polylines):
    convex_hull_polygon = find_largest_convex_hull(polylines)
    return convex_hull_polygon


def find_largest_convex_hull(polylines):
    # Merge all points from the polylines into a single collection
    all_points = []
    for polyline in polylines:
        all_points.extend(list(polyline.coords))
    
    # Convert the points to a NumPy array for processing with scipy
    points_array = np.array(all_points)
    
    # Compute the convex hull using scipy
    hull = ConvexHull(points_array)
    
    # Extract the vertices of the convex hull
    hull_points = points_array[hull.vertices]
    
    # Create a shapely Polygon object representing the convex hull
    convex_hull_polygon = LineString(hull_points).convex_hull
    
    return convex_hull_polygon


def project_point_onto_line(point, line):
    """Project a point onto a line segment and return the projected point."""
    line_start, line_end = np.array(line.coords[0]), np.array(line.coords[1])
    line_vec = line_end - line_start
    point_vec = np.array(point.coords[0]) - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)    
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t * line_vec
    return Point(nearest)


def find_nearest_projection_on_polyline(point, polyline):
    """Find the nearest projected point of a point onto a polyline."""
    min_dist = float('inf')
    nearest_point = None
    for i in range(len(polyline.coords) - 1):
        segment = LineString(polyline.coords[i:i+2])
        proj_point = project_point_onto_line(point, segment)
        dist = point.distance(proj_point)
        if dist < min_dist:
            min_dist = dist
            nearest_point = proj_point
    return np.array(nearest_point.coords)


def find_and_sort_intersections(segmenet1, segment2):
    # Convert polylines to LineString objects

    # Find the intersection between the two LineStrings
    intersection = segmenet1.intersection(segment2)
    # import ipdb; ipdb.set_trace()
    # Prepare a list to store intersection points
    intersections = []

    # Check the type of intersection
    if intersection.geom_type == "MultiPoint":
        intersections.extend(list(intersection.geoms))  # 正确获取 MultiPoint 的每个点
    elif intersection.geom_type == "Point":
        intersections.append(intersection)
    elif intersection.geom_type == "LineString":
        bound = intersection.boundary
        # import ipdb;ipdb.set_trace()
        if bound.geom_type == "MultiPoint":
            intersections.extend(list(bound.geoms))  # 正确获取 MultiPoint 的每个点
        else:
            intersections.append(bound)
        # intersections.extend(list(intersection.boundary))
    elif intersection.geom_type == "MultiLineString":
        for line in intersection.geoms:
            bound = line.boundary
            if bound.geom_type == "MultiPoint":
                intersections.extend(list(bound.geoms))  # 正确获取 MultiPoint 的每个点
            else:
                intersections.append(bound)
            # intersections.extend(list(line.boundary))

    # Remove duplicates and ensure they are Point objects
    
    unique_intersections = [Point(coords) for coords in set(pt.coords[0] for pt in intersections)]
    # import ipdb; ipdb.set_trace()
    # Sort the intersection points by their distance along the first polyline
    sorted_intersections = sorted(unique_intersections, key=lambda pt: segmenet1.project(pt))
    # import ipdb; ipdb.set_trace()
    return sorted_intersections


def get_intersection_point_on_line(line, intersection):
    intersection_points  = find_and_sort_intersections(LineString(line), intersection)
    if len(intersection_points) >= 2:
        line_intersect_start = intersection_points[0]
        line_intersect_end = intersection_points[-1]
    elif len(intersection_points) == 1:
        if intersection.contains(Point(line[0])):
            line_intersect_start = Point(line[0])
            line_intersect_end = intersection_points[0]
        elif intersection.contains(Point(line[-1])):
            line_intersect_start = Point(line[-1])
            line_intersect_end = intersection_points[0]
        else:
            return None, None            
    else:
        return None, None            
    return line_intersect_start, line_intersect_end

def merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end):
    # get nearest point on line2 to line2_intersect_start
    line2_point_to_merge = []
    line2_intersect_start_dis = line2.project(line2_intersect_start)
    line2_intersect_end_dis = line2.project(line2_intersect_end)
    for point in np.array(line2.coords):
        point_geom = Point(point)
        dis = line2.project(point_geom)
        if dis > line2_intersect_start_dis and dis < line2_intersect_end_dis:
            line2_point_to_merge.append(point)
            
    # merged the points
    merged_line2_points = []
    for point in line2_point_to_merge:
        # Use the `project` method to find the distance along the polyline to the closest point
        point_geom = Point(point)
        # Use the `interpolate` method to find the actual point on the polyline
        closest_point_on_line = find_nearest_projection_on_polyline(point_geom, line1)
        if len(closest_point_on_line) == 0:
            merged_line2_points.append(point)
        else:
            merged_line2_points.append(((closest_point_on_line + point) / 2)[0])

    if len(merged_line2_points) == 0:
        merged_line2_points = np.array([]).reshape(0, 2)
    else:
        merged_line2_points = np.array(merged_line2_points)
        
    return merged_line2_points        

def segment_line_based_on_merged_area(line, merged_points):
    
    if len(merged_points) == 0:
        return  np.array(line.coords),  np.array([]).reshape(0, 2)
    
    first_merged_point = merged_points[0]
    last_merged_point = merged_points[-1]
    
    start_dis = line.project(Point(first_merged_point))
    end_dis = line.project(Point(last_merged_point))
    
    start_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) < start_dis:
            start_segmenet.append(point)
    
    end_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) > end_dis:
            end_segmenet.append(point)
            
    if len(start_segmenet) == 0:
        start_segmenet = np.array([]).reshape(0, 2)
    else:
        start_segmenet = np.array(start_segmenet)
        
    if len(end_segmenet) == 0:
        end_segmenet = np.array([]).reshape(0, 2)
    else:
        end_segmenet = np.array(end_segmenet)
    
    return start_segmenet, end_segmenet
    
def get_bbox_size_for_points(points):
    if len(points) == 0:
        return 0, 0
    
    # Initialize min and max coordinates with the first point
    min_x, min_y = points[0]
    max_x, max_y = points[0]

    # Iterate through each point to update min and max coordinates
    for x, y in points[1:]:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return max_x - min_x, max_y - min_y

def get_longer_segmenent_to_merged_points(l1_segment, l2_segment, merged_line2_points, segment_type="start"):
    # remove points from segments if it's too close to merged_line2_points
    l1_segment_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l1_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l1_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 0:
        l1_segment_temp = l1_segment
        
                
    l1_segment = np.array(l1_segment_temp)
    
    l2_segmenet_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l2_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l2_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 0:
        l2_segmenet_temp = l2_segment
                
    l2_segment = np.array(l2_segmenet_temp)
    
    if segment_type == "start":
        
        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        
        l1_start_box_size = get_bbox_size_for_points(temp)
        
        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        l2_start_box_size = get_bbox_size_for_points(temp)
    
        if l2_start_box_size[0]*l2_start_box_size[1] >= l1_start_box_size[0]*l1_start_box_size[1]:
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment
    else:
        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l1_end_box_size = get_bbox_size_for_points(temp)
        
        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l2_end_box_size = get_bbox_size_for_points(temp)
    
        if l2_end_box_size[0]*l2_end_box_size[1] >= l1_end_box_size[0]*l1_end_box_size[1]:
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment
    
    if len(longer_segment) == 0:
        longer_segment = np.array([]).reshape(0, 2)
    else:
        longer_segment = np.array(longer_segment)
        
    return longer_segment
    
def get_line_lineList_max_intersection(merged_lines, line, thickness=4):
    pre_line = merged_lines[-1]
    max_iou = 0
    merged_line_index = 0
    for line_index, one_merged_line in enumerate(merged_lines):
        line1 = LineString(one_merged_line)
        line2 = LineString(line)
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        if intersection.area / thick_line2.area > max_iou:
            max_iou = intersection.area / thick_line2.area
            pre_line = np.array(line1.coords)
            merged_line_index = line_index
    return intersection, pre_line, merged_line_index
    
def algin_l2_with_l1(line1, line2):
    
    if len(line1) > len(line2):
        l2_len = len(line2)
        line1_geom = LineString(line1)
        interval_length = line1_geom.length / (l2_len - 1)
        line1 = np.array([[line1_geom.interpolate(interval_length * i).x,\
                                         line1_geom.interpolate(interval_length * i).y]\
                                                    for i in range(l2_len)])
        # line1 = [np.array(line1_geom.interpolate(interval_length * i)) for i in range(l2_len)]
        
    elif len(line1) < len(line2):
        l1_len = len(line1)
        line2_geom = LineString(line2)
        interval_length = line2_geom.length / (l1_len - 1)
        line2 = np.array([[line2_geom.interpolate(interval_length * i).x,\
                                         line2_geom.interpolate(interval_length * i).y]\
                                                    for i in range(l1_len)])
        # line2 = [np.array(line2_geom.interpolate(interval_length * i)) for i in range(l1_len)]
    
    # make line1 and line2 same direction, pre_line.coords[0] shold be closer to line2.coords[0]
    
    if isinstance(line1,list):
        # import ipdb; ipdb.set_trace()
        print(line1)
    line1_geom = LineString(line1)
    line2_flip = np.flip(line2, axis=0)
    # import ipdb; ipdb.set_trace()
    
    line2_traj_len = 0
    for point_idx, point in enumerate(line2):
        line2_traj_len += np.linalg.norm(point - line1[point_idx])
    
    flip_line2_traj_len = 0
    for point_idx, point in enumerate(line2_flip):
        flip_line2_traj_len += np.linalg.norm(point - line1[point_idx])
    
        
    if abs(flip_line2_traj_len - line2_traj_len) < 3:
        # get the trajectory length
        line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                line2_walk_len += line1_geom.project(Point(proj_point[0]))
        
        flip_line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                flip_line2_walk_len += line1_geom.project(Point(proj_point[0]))
        
        if flip_line2_walk_len < line2_walk_len:
            return line2_flip
        else:
            return line2
        
    
    if flip_line2_traj_len < line2_traj_len:
        return line2_flip
    else:
        return line2

def _is_u_shape(line, direction):
    assert direction in ['left', 'right'], 'Wrong direction argument {}'.format(direction)
    line_geom = LineString(line)
    length = line_geom.length
    mid_point = np.array(line_geom.interpolate(length / 2).coords)[0]
    start = line[0]
    end = line[-1]

    if direction == 'left':
        cond1 = mid_point[0] < start[0] and mid_point[0] < end[0]
    else:
        cond1 = mid_point[0] > start[0] and mid_point[0] > end[0]
    
    dist_start_end = np.sqrt((start[0] - end[0])**2 + (start[1]-end[1])**2)
    cond2 = length >= math.pi / 2 * dist_start_end

    return cond1 and cond2

def check_circle(pre_line, vec):

    # if the last line in merged_lines is a circle
    if np.linalg.norm(pre_line[0] - pre_line[-1]) == 0:
        return True
    
    # if the last line in merged_lines is almost a circle and the new line is close to the circle
    if np.linalg.norm(pre_line[0] - pre_line[-1]) < 0.1:
        vec_2_circle_distance = 0
        for point in vec:
            vec_2_circle_distance += LineString(pre_line).distance(Point(point))
        if vec_2_circle_distance < 3:
            return True
    return False
        
def connect_polygon(merged_polyline, merged_lines):
    start_end_connect = [merged_polyline[0], merged_polyline[-1]]
    iou = []
    length_ratio = []
    for one_merged_line in merged_lines:
        line1 = LineString(one_merged_line)
        line2 = LineString(start_end_connect)
        thickness = 1
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        iou.append(intersection.area / thick_line2.area)
        length_ratio.append(line1.length / line2.length)

    if max(iou) > 0.95 and max(length_ratio) > 3.0:
        merged_polyline = np.concatenate((merged_polyline, [merged_polyline[0]]), axis=0)
    return merged_polyline
    
def iou_merge_boundry(merged_lines, vec, thickness=1):

    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(merged_lines, vec, thickness)

    # corner case: check if the last line in merged_lines is a circle
    if check_circle(pre_line, vec):
        return merged_lines

    # Handle U-shape, the main corner case
    if _is_u_shape(pre_line, 'left'):
        if _is_u_shape(vec, 'right'):
            # Two u shapes with opposite directions, directly generate a polygon exterior
            polygon = find_largest_convex_hull([LineString(pre_line), LineString(vec)])
            merged_lines[-1] = np.array(polygon.exterior.coords)
            return merged_lines
        elif not _is_u_shape(vec, 'left'):
            line_geom1 = LineString(pre_line)
            line1_dists = np.array([line_geom1.project(Point(x)) for x in pre_line])
            split_mask = line1_dists > line_geom1.length / 2
            split_1 = LineString(pre_line[~split_mask])
            split_2 = LineString(pre_line[split_mask])

            # get the projected distance
            np1 = np.array(nearest_points(split_1, Point(Point(pre_line[-1])))[0].coords)[0]
            np2 = np.array(nearest_points(split_2, Point(Point(pre_line[0])))[0].coords)[0]
            dist1 = np.linalg.norm(np1-pre_line[-1])
            dist2 = np.linalg.norm(np2-pre_line[0])
            dist = min(dist1, dist2)

            if dist < thickness:
                line_geom2 = LineString(vec)
                dist1 = line_geom2.distance(Point(pre_line[0]))
                dist2 = line_geom2.distance(Point(pre_line[-1]))
                pt = pre_line[0] if dist1 <= dist2 else pre_line[-1]
                if vec[0][0] > vec[1][0]:
                    vec = np.array(vec[::-1])
                    line_geom2 = LineString(vec)
                proj_length = line_geom2.project(Point(pt))
                l2_select_mask = np.array([line_geom2.project(Point(x)) > proj_length for x in vec])
                selected_l2 = vec[l2_select_mask]
                merged_result = np.concatenate([pre_line[:-1, :], pt[None, ...], selected_l2], axis=0)
                merged_lines[-1] = merged_result
                return merged_lines
    
    # align the new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)
    line1 = LineString(pre_line)
    line2 = LineString(vec)
    
    # get the intersection points between IOU area and two lines
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(pre_line, intersection)
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(vec, intersection)
    
    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if line1_intersect_start is None or line1_intersect_end is None or line2_intersect_start is None or line2_intersect_end is None:
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])
    
    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end)
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(line2, line1, line1_intersect_start, line1_intersect_end)
    
    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(line2, merged_line2_points)
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(line1, merged_line1_points)
    
    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(l1_start_segment, l2_start_segment, merged_line2_points, segment_type="start")
    end_segment = get_longer_segmenent_to_merged_points(l1_end_segment, l2_end_segment, merged_line2_points, segment_type="end")
    merged_polyline = np.concatenate((start_segment, merged_line2_points, end_segment), axis=0)
    
    # corner case : check if need to connect the polyline to form a circle
    merged_polyline = connect_polygon(merged_polyline, merged_lines)
    
    merged_lines[merged_line_index] = merged_polyline
  
    return merged_lines

def iou_merge_divider(merged_lines, vec, thickness=1):
    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    # pre_line : the line in merged_lines that has max IOU with the new line
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(merged_lines, vec, thickness)
    # align the new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)
    
    line1 = LineString(pre_line)
    line2 = LineString(vec)
    
    # get the intersection points between IOU area and two lines
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(pre_line, intersection)
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(vec, intersection)
    
    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if line1_intersect_start is None or line1_intersect_end is None or line2_intersect_start is None or line2_intersect_end is None:
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])
    
    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end)
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(line2, line1, line1_intersect_start, line1_intersect_end)
    
    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(line2, merged_line2_points)
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(line1, merged_line1_points)
    
    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(l1_start_segment, l2_start_segment, merged_line2_points, segment_type="start")
    end_segment = get_longer_segmenent_to_merged_points(l1_end_segment, l2_end_segment, merged_line2_points, segment_type="end")
    merged_polyline = np.concatenate((start_segment, merged_line2_points, end_segment), axis=0)
    
    # update the merged_lines
    merged_lines[merged_line_index] = merged_polyline
    
    return merged_lines

def merge_divider(vecs=None, thickness=1):
    merged_lines = []
    for vec in vecs:
        
        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue
        
        # thicken the vec (the new line) and the merged_lines calculate the max IOU between the new line and the merged_lines
        iou = []
        for one_merged_line in merged_lines:
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)
        
        # If the max IOU is 0, add the new line to the merged_lines
        if max(iou) == 0:
            merged_lines.append(vec)
        # If IOU is not 0, merge the new line with the line in the merged_lines
        else:
            merged_lines = iou_merge_divider(merged_lines, vec, thickness=thickness)

           
    return merged_lines

def merge_boundary(vecs=None,tag = None, thickness=1, iou_threshold=0.95):
    merged_lines = []
    # import ipdb;ipdb.set_trace()
    # i = 0
    for vec in vecs:
        # i += 1

        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue
        
        # thicken the vec (the new line) and the merged_lines calculate the max IOU between the new line and the merged_lines
        iou = []
        for one_merged_line in merged_lines:
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)
        
        # If the max IOU larger than the threshold, skip the new line
        if max(iou) > iou_threshold:
            continue

        # If IOU is not 0, merge the new line with the line in the merged_lines
        if max(iou) > 0:
            # if tag == '2_1494' and i == 4:
            #     import ipdb;ipdb.set_trace()
            merged_lines = iou_merge_boundry(merged_lines, vec, thickness=thickness)
        else:
            merged_lines.append(vec)
    # import ipdb;ipdb.set_trace()
    return merged_lines


def get_prev2curr_vectors(vecs=None, prev2curr_matrix=None, origin=None, roi_size=None, denormalize=False, clip=False):
    # transform prev vectors
    if vecs is not None and len(vecs) > 0:
        vecs = np.stack(vecs, 0)
        N, num_points, dim = vecs.shape
        # if denormalize:
        #     denormed_vecs = vecs * roi_size + origin  # (num_prop, num_pts, 2)
        # else:
        #     denormed_vecs = vecs
        denormed_vecs = vecs

        if dim == 2:
            denormed_vecs = np.concatenate([
                denormed_vecs,
                np.zeros((N, num_points, 1)),  # z-axis
                np.ones((N, num_points, 1))  # 4-th dim
            ], axis=-1)  # (num_prop, num_pts, 4)
        else:
            denormed_vecs = np.concatenate([
                denormed_vecs[...,:2],
                np.zeros((N, num_points, 1)),
                np.ones((N, num_points, 1))  # 4-th dim
            ], axis=-1)  # (num_prop, num_pts, 4)

        # Note: np.einsum does not support broadcasting over different dtypes, so we ensure
        # both arrays are of the same dtype before the operation.
        transformed_vecs = np.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_vecs.astype(np.double))
        transformed_vecs = transformed_vecs.astype(np.float32)
        # vecs = (transformed_vecs[..., :2] - origin) / roi_size  # (num_prop, num_pts, 2)
        vecs = transformed_vecs[..., :2]
        if clip:
            vecs = np.clip(vecs, a_min=0., a_max=1.)
        # vecs = vecs * roi_size + origin

    return vecs