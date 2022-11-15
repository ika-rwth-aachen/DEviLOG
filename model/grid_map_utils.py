# ==============================================================================
# MIT License
#
# Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


from typing import List
import numpy as np
import tensorflow as tf
import math
import cv2
from skimage.draw import line


def evidencesToMasses(logits: tf.Tensor) -> tf.Tensor:
    """Convert evidences in range (0,inf) into belief masses in range (0,1) using Subjective Logic

    Args:
        logits (tf.Tensor): Evidences of shape (..., num_classes)

    Returns:
        prob (tf.Tensor): Belief masses of shape (..., num_classes)
        u (tf.Tensor): Uncertainty masses of shape  (..., 1)
        S (tf.Tensor): Dirichlet Strengths of shape (..., 1)
        num_classes (tf.int): Number of classes, e.g. 2 for free/occupied
    """

    logits = tf.cast(logits, tf.float32)
    num_classes = tf.cast(tf.shape(logits)[-1], tf.dtypes.float32)

    # convert evidences (y_pred) to parameters of Dirichlet distribution (alpha)
    alpha = tf.math.add(logits, tf.ones(tf.shape(logits)), name="ogm/params")

    # Dirichlet strength (sum alpha for all classes)
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True, name="ogm/S")

    u = tf.math.divide(num_classes, S, name="ogm/uncertainty_mass")
    prob = tf.math.divide(logits, S, name="ogm/belief_masses")

    return prob, u, S, num_classes


def gridmapToImage(grid_map: tf.Tensor) -> tf.Tensor:
    """Create images from occupancy grid maps

    Grid map can have two (free/occupied) or three channels (occupied_static/free/occupied_dynamic) `K`.

    Args:
        grid_map (tf.Tensor): Occupancy grid map with belief masses of shape (W, L, K) or (B, W, L, K)

    Returns:
        tf.Tensor: RGB image visualizing occupancy grid maps
    """

    image = tf.cast(255.0 * grid_map, dtype=tf.uint8)

    if image.shape[-1] == 3:
        # tensor contains m_occupied_static=red/m_free=green/m_occupied_dynamic=blue
        pass
    else:
        # tensor contains m_free=green/m_occupied=red
        blue_channel = tf.zeros_like(image[..., 0])
        green_channel, red_channel = tf.unstack(image, axis=-1)
        image = tf.stack((red_channel, green_channel, blue_channel), axis=-1)

    return tf.cast(image, dtype=tf.uint8)


class GridConfig:

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, step_size_x,
                 step_size_y) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.step_size_x = step_size_x
        self.step_size_y = step_size_y

        if None not in [
                self.x_max, self.x_min, self.y_max, self.y_min, self.step_size_x, self.step_size_y
        ]:
            self.grid_size = (int((self.x_max - self.x_min) / self.step_size_x), 
                              int((self.y_max - self.y_min) / self.step_size_y))
            self.cell_size = ((self.x_max - self.x_min) / self.grid_size[0],
                              (self.y_max - self.y_min) / self.grid_size[1])


def precisionRecall(predicted_grid_map,
                    label_grid_map,
                    threshold=0.5,
                    sample_weights=None):
    precisions = []
    recalls = []
    ious = []

    if sample_weights is None:
        sample_weights = 1.0

    if predicted_grid_map.shape[-1] == 3:
        # create combined "occupied" layer that combines "static" and "dynamic"
        predicted_grid_map_occ = tf.expand_dims(
            tf.math.add(predicted_grid_map[..., 0], predicted_grid_map[..., 2]),
            -1)
        predicted_grid_map = tf.concat(
            (predicted_grid_map, predicted_grid_map_occ), axis=-1)

        label_grid_map_occ = tf.expand_dims(
            tf.math.add(label_grid_map[..., 0], label_grid_map[..., 2]), -1)
        label_grid_map = tf.concat((label_grid_map, label_grid_map_occ),
                                   axis=-1)

    for channel in range(predicted_grid_map.shape[-1]):
        tp = tf.math.reduce_sum(
            tf.where(
                tf.logical_and(predicted_grid_map[..., channel] > threshold,
                               label_grid_map[..., channel] > threshold),
                sample_weights, 0.0))
        fp = tf.math.reduce_sum(
            tf.where(
                tf.logical_and(predicted_grid_map[..., channel] > threshold,
                               label_grid_map[..., channel] <= threshold),
                sample_weights, 0.0))
        fn = tf.math.reduce_sum(
            tf.where(
                tf.logical_and(predicted_grid_map[..., channel] <= threshold,
                               label_grid_map[..., channel] > threshold),
                sample_weights, 0.0))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        iou = tp / (tp + fn + fp)

        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)

    return precisions, recalls, ious


def errorMap(predicted_grid_map, label_grid_map, threshold=0.5):

    fp_map = []
    fn_map = []
    tp_map = []

    # use only cells with known state in label for evaluation
    sample_weights = tf.where(tf.reduce_any(label_grid_map > 0.5, axis=-1), 1.0,
                              0.0)

    for channel in range(predicted_grid_map.shape[-1]):
        fp = tf.where(
            tf.logical_and(predicted_grid_map[..., channel] > threshold,
                           label_grid_map[..., channel] <= threshold),
            sample_weights, 0.0)
        fp_map.append(tf.expand_dims(fp, -1))

        fn = tf.where(
            tf.logical_and(predicted_grid_map[..., channel] <= threshold,
                           label_grid_map[..., channel] > threshold),
            sample_weights, 0.0)
        fn_map.append(tf.expand_dims(fn, -1))

        tp = tf.where(
            tf.logical_and(predicted_grid_map[..., channel] > threshold,
                           label_grid_map[..., channel] > threshold),
            sample_weights, 0.0)
        tp_map.append(tf.expand_dims(tp, -1))

    # ensure to create valid RGB image
    if tf.shape(predicted_grid_map)[-1] == 3:
        fp_map = tf.concat(fp_map, axis=-1)
        fn_map = tf.concat(fn_map, axis=-1)
        tp_map = tf.concat(tp_map, axis=-1)
    elif tf.shape(predicted_grid_map)[-1] == 2:
        fp_map = tf.concat([fp_map[1], fp_map[0], tf.zeros_like(fp_map[0])], axis=-1)
        fn_map = tf.concat([fn_map[1], fn_map[0], tf.zeros_like(fn_map[0])], axis=-1)
        tp_map = tf.concat([tp_map[1], tp_map[0], tf.zeros_like(tp_map[0])], axis=-1)
    else:
        raise ValueError(f"Only grid maps with 2 or 3 channels are supported but it has {tf.shape(predicted_grid_map)[-1]}.")

    return fp_map, fn_map, tp_map


def createObjectImage(object_list, img, cell_length, cell_width, min_points_in_bbox=5):
    height = tf.shape(img)[-2]
    width = tf.shape(img)[-1]
    center_x = height//2
    center_y = width//2

    if tf.shape(object_list)[-2] > 0:
        # ignore object classes (9 = movable_object)
        objects_class = object_list[..., 7]
        objects_lidar_pts = object_list[..., 8]
        mask_classes = tf.logical_and(tf.logical_and(tf.logical_and(tf.logical_and(objects_class != 9, objects_class != 10), 
            objects_class != 11), objects_class != 12), objects_lidar_pts >= min_points_in_bbox)
        object_list = tf.boolean_mask(object_list, mask_classes)

        objects_x = object_list[..., 0]
        objects_y = object_list[..., 1]
        objects_yaw = object_list[..., 3]
        objects_length = object_list[..., 4]
        objects_width = object_list[..., 5]

        # calculate corners of boxes (x points upwards, y points left)
        x_1 = objects_x - objects_length/2*tf.math.cos(objects_yaw) + objects_width/2*tf.math.sin(objects_yaw)
        y_1 = objects_y - objects_length/2*tf.math.sin(objects_yaw) - objects_width/2*tf.math.cos(objects_yaw)
        x_2 = objects_x - objects_length/2*tf.math.cos(objects_yaw) - objects_width/2*tf.math.sin(objects_yaw)
        y_2 = objects_y - objects_length/2*tf.math.sin(objects_yaw) + objects_width/2*tf.math.cos(objects_yaw)
        x_3 = objects_x + objects_length/2*tf.math.cos(objects_yaw) - objects_width/2*tf.math.sin(objects_yaw)
        y_3 = objects_y + objects_length/2*tf.math.sin(objects_yaw) + objects_width/2*tf.math.cos(objects_yaw)
        x_4 = objects_x + objects_length/2*tf.math.cos(objects_yaw) + objects_width/2*tf.math.sin(objects_yaw)
        y_4 = objects_y + objects_length/2*tf.math.sin(objects_yaw) - objects_width/2*tf.math.cos(objects_yaw)

        # convert from meters to cells indices
        x_1 = center_x + tf.cast(x_1 / cell_length, tf.int32)
        y_1 = center_y + tf.cast(y_1 / cell_width, tf.int32)
        x_2 = center_x + tf.cast(x_2 / cell_length, tf.int32)
        y_2 = center_y + tf.cast(y_2 / cell_width, tf.int32)
        x_3 = center_x + tf.cast(x_3 / cell_length, tf.int32)
        y_3 = center_y + tf.cast(y_3 / cell_width, tf.int32)
        x_4 = center_x + tf.cast(x_4 / cell_length, tf.int32)
        y_4 = center_y + tf.cast(y_4 / cell_width, tf.int32)
        point_1 = tf.stack([y_1, x_1], axis=-1)
        point_2 = tf.stack([y_2, x_2], axis=-1)
        point_3 = tf.stack([y_3, x_3], axis=-1)
        point_4 = tf.stack([y_4, x_4], axis=-1)
        points = tf.stack([point_1, point_2, point_3, point_4], axis=1)

        # create one grid layer per object
        img = tf.numpy_function(lambda img, points: cv2.drawContours(img, np.int0(points), -1, 1.0, -1), inp=[img, points], Tout=tf.float32)
        img = tf.reverse(img, (0, 1))  # to make x pointing downwards and y pointing right as required for images

    return img


def createVisibilityMap(ogm_static, ogm_dynamic, grid_config, use_raycasting = False, min_distance = [0, 0]):

    cells_x = tf.shape(ogm_static)[0]
    cells_y = tf.shape(ogm_static)[1]
    center_x = cells_x//2
    center_y = cells_y//2

    if use_raycasting:
        visibility_map = tf.zeros([cells_x, cells_y], dtype=bool)
        # cast ray from center to all border cells and set all cells in visiblity_map = 1 until an occupied cell is reached
        border_x = tf.concat([tf.range(cells_x), tf.repeat(tf.constant([0]), cells_y), tf.range(cells_x), tf.repeat(cells_x-1, cells_y)], axis=0)
        border_y = tf.concat([tf.repeat(tf.constant([0]), cells_x), tf.range(cells_y), tf.repeat(cells_y-1, cells_x), tf.range(cells_y)], axis=0)
        center_x = tf.repeat(center_x, tf.shape(border_x)[0])
        center_y = tf.repeat(center_y, tf.shape(border_x)[0])

        ogm_occupied = tf.stack([ogm_static > 0.5, ogm_dynamic > 0.5], axis=-1)
        occupied_map = tf.reduce_any(ogm_occupied, axis=-1)

        def castRay(inp):
            center_x, center_y, border_x, border_y = inp

            def castRay(start_x, start_y, end_x, end_y, visibility_map, occupied_map):
                rr, cc = line(start_x, start_y, end_x, end_y)
                for (r, c) in zip(rr, cc):
                    if occupied_map[r, c]:
                        # cell is occupied --> visibility ends
                        visibility_map[r, c] = 0
                        break
                    else:
                        visibility_map[r, c] = 1
                return visibility_map
            
            return tf.numpy_function(castRay, inp=[center_x, center_y, border_x, border_y, visibility_map, occupied_map], Tout=tf.bool)

        visibility_map = tf.map_fn(castRay, elems=(center_x, center_y, border_x, border_y), fn_output_signature=tf.bool, parallel_iterations=100)
        visibility_map = tf.reduce_any(visibility_map, axis=0)
    else:
        visibility_map = tf.ones([cells_x, cells_y], dtype=bool)
    
    def distanceVisibility(visibility_map, center_x, center_y, cell_size_x, cell_size_y, min_distance):
        # remove below minimum distance
        cells_x = int(min_distance[0]/cell_size_x)
        cells_y = int(min_distance[1]/cell_size_y)
        visibility_map[center_x-cells_x:center_x+cells_x, center_y-cells_y:center_y+cells_y] = 0

        return visibility_map
        
    # remove visibility below min and above max distance
    if min_distance != [0, 0]:
        visibility_map = tf.numpy_function(distanceVisibility,
                        inp=[visibility_map, cells_x//2, cells_y//2, grid_config.cell_size[0], grid_config.cell_size[1], min_distance], Tout=tf.bool)

    return visibility_map


def mapToOgm(label_map, label_objects, grid_config: GridConfig, combine_occupied: bool = False, use_raycasting: bool = False, 
    min_points_in_bbox: int = 5, min_distance: List[float] = [0, 0]):

    # crop OGM from larger nuScenes map (label_map size: 100mx100m, cell_size: 0.2mx0.2m)
    height = (grid_config.x_max - grid_config.x_min)/0.2
    width = (grid_config.y_max - grid_config.y_min)/0.2
    label_map = tf.cast(label_map[int(label_map.shape[0]/2 - height/2):int(label_map.shape[0]/2 + height/2), 
                                  int(label_map.shape[1]/2 - width/2):int(label_map.shape[1]/2 + width/2)], tf.float32)
    # resize to dimensions of grid map
    ogm_free = tf.image.resize(tf.expand_dims(label_map, -1), grid_config.grid_size)
    ogm_free = ogm_free[..., 0]
    # create dynamic layer from object list
    ogm_dynamic = tf.zeros_like(ogm_free)
    ogm_dynamic = tf.numpy_function(createObjectImage, inp=[label_objects, ogm_dynamic, grid_config.step_size_x, grid_config.step_size_y, min_points_in_bbox], Tout=tf.float32)
    # remove free where dynamic, set all other cells static
    ogm_free = tf.where(ogm_dynamic > 0.5, 0.0, ogm_free)
    ogm_static = tf.where(tf.logical_and(ogm_free < 0.5, ogm_dynamic < 0.5), 1.0, 0.0)

    # filter label grid map using raycasting and/or minimum/maximum distance
    visibility_map = createVisibilityMap(ogm_static, ogm_dynamic, grid_config, use_raycasting, min_distance)
    ogm_static = tf.where(visibility_map, ogm_static, 0.0)
    ogm_free = tf.where(visibility_map, ogm_free, 0.0)
    # ogm_dynamic = tf.where(visibility_map, ogm_dynamic, 0.0)

    # combine layers to grid map
    if combine_occupied:
        ogm = tf.stack([ogm_free, tf.clip_by_value(tf.add(ogm_static, ogm_dynamic), 0.0, 1.0)], axis=-1)
    else:
        ogm = tf.stack([ogm_static, ogm_free, ogm_dynamic], axis=-1)

    return ogm
