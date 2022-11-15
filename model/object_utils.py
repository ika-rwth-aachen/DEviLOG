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

import csv
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from point_pillars import createPillarsTarget

# attribute indices of common object representation
I_CLASS = 0
I_X = 1
I_Y = 2
I_Z = 3
I_YAW = 4
I_LENGTH = 5
I_WIDTH = 6
I_HEIGHT = 7
I_X_STD = 8
I_Y_STD = 9
I_YAW_STD = 10

def readObjectList(objects: tf.Tensor, label_type: str = 'demo'):
    """Create harmonized object representation from different datasets.

    Args:
        objects: list of N objects with F features in shape [..., N, F]
        label_type (str, optional): Dataset where objects originates. Defaults to 'demo'.

    Returns:
        np.ndarray: array of objects with attributes (class_id, x, y, z, yaw, length, width, height)
    """
    if label_type == 'demo':
        i_class = 2
        i_x = 4
        i_y = 5
        i_z = 6
        i_yaw = 9
        i_dx = 12
        i_dy = 11
        i_dz = 13
    elif label_type == 'nuscenes':
        i_class = 0
        i_x = 1
        i_y = 2
        i_z = 3
        i_yaw = 4
        i_dx = 5
        i_dy = 6
        i_dz = 7
    else:
        raise ValueError(f"Object label type {label_type} is not supported")
    x = objects[..., i_x]
    y = objects[..., i_y]
    z = objects[..., i_z]
    yaw = objects[..., i_yaw]
    length = objects[..., i_dx]
    width = objects[..., i_dy]
    objects_height = objects[..., i_dz]
    objects_class = tf.cast(tf.math.round(objects[..., i_class]), tf.uint8)

    # ensure that yaw is in range [-pi, pi]
    yaw = reduceAngleRange(yaw)
    objects_pose = tf.stack([x, y, z, yaw], axis=-1)
    objects_dimensions = tf.stack([length, width, objects_height], axis=-1)

    return objects_class, objects_pose, objects_dimensions


def readObjectsFromCsv(file: Union[str, np.ndarray]):

    if isinstance(file, np.ndarray):
        file = file[()]

    # parse csv-file
    with open(file) as f:
        reader = csv.reader(f, delimiter=",")
        _ = next(reader)  # header
        object_list = []
        for object in reader:
            object_list.append(object)
    object_list = np.asarray(object_list, dtype=np.float32)
    return object_list


def rotateObjectList(objects_pose: tf.Tensor, angle: float) -> tf.Tensor:
    """Rotate object centroids by given angle around z axis.

    Args:
        objects_pose: Object centroids (x, y, z, yaw) in shape [N, 4]
        angle: Rotation angle in radians

    Returns:
        tf.Tensor: Rotated centroids in shape [N, 4]
    """

    objects_pose = tf.ensure_shape(objects_pose, [None, 4])

    # yapf: disable
    rotation = tf.constant([[math.cos(angle), math.sin(angle), 0],
                            [-math.sin(angle), math.cos(angle), 0],
                            [0, 0, 1]])
    # yapf: enable
    objects_xyz = objects_pose[..., 0:3]
    objects_xyz = tf.linalg.matmul(objects_xyz, rotation)

    yaw = objects_pose[..., 3:4] + angle

    yaw = reduceAngleRange(yaw)
    objects_pose = tf.concat([objects_xyz, yaw], axis=-1)

    return objects_pose


def reduceAngleRange(angles: tf.Tensor) -> tf.Tensor:
    """Ensure that angles are in range [-pi, pi]

    Args:
        angles (tf.Tensor): tensor with angles of arbitrary shape

    Returns:
        tf.Tensor: same tensor with angles in range [-pi, pi]
    """
    angles = tf.math.floormod(angles, 2 * np.pi)
    angles = tf.where(angles > np.pi, angles - 2 * np.pi, angles)
    angles = tf.where(angles < -1 * np.pi, angles + 2 * np.pi, angles)
    return angles


def createPointPillarsLabels(
    objects_class: tf.Tensor, objects_pose: tf.Tensor,
    objects_dimensions: tf.Tensor, anchors: List[float],
    label_class_names: List[str], classes: Dict[str, int], x_min: float,
    x_max: float, y_min: float, y_max: float, z_min: float, z_max: float,
    step_x_size: float, step_y_size: float, downscaling_factor: int,
    pos_iou_threshold: float, neg_iou_threshold: float, angle_threshold: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Create labels for PointPillars network from object list.

    Args:
        objects_class (tf.Tensor): list of object classes in shape [num_objects]
        objects_pose (tf.Tensor): list of object poses (x, y, z, yaw) in shape [num_objects, 4]
        objects_dimensions (tf.Tensor): list of object dimensions (length, width, height) in shape [num_objects, 3]
        anchors (List[float]): list of anchors (length, width, height, z, yaw)
        label_class_names (List[str]): List of class names in order of class ids in training data
        classes (Dict[str, int]): mapping of class names to ids for classes to be used for training
        x_min (float): minimum x coordinate of points to be processed
        x_max (float): maximum x coordinate of points to be processed
        y_min (float): minimum y coordinate of points to be processed
        y_max (float): maximum y coordinate of points to be processed
        z_min (float): minimum z coordinate of points to be processed
        z_max (float): maximum z coordinate of points to be processed
        step_x_size (float): size of one pillar in x dimension
        step_y_size (float): size of one pillar in y dimension
        downscaling_factor (int): downscaling of pillar map to network output
        pos_iou_threshold (float): IoU threshold for positives
        neg_iou_threshold (float): IoU threshold for negatives
        angle_threshold (float): angle threshold

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: network labels
    """

    anchors = np.array(anchors, dtype=np.float32)
    if len(anchors.shape) != 2 or anchors.shape[1] != 5:
        raise ValueError(
            f"Expected anchors to have shape [num_anchors, 5] but has {tf.shape(anchors)}"
        )
    anchor_dims = anchors[:, 0:3]
    anchor_z = anchors[:, 3]
    anchor_yaw = anchors[:, 4]

    def make_ground_truth(objects_class: np.ndarray, objects_pose: np.ndarray,
                          objects_dimensions: np.ndarray):
        # assign new class ids according to selected classes
        mask = []
        for i in range(len(objects_class)):
            class_id = int(objects_class[i].item())
            class_name = label_class_names[class_id]
            if class_name in classes:
                mask.append(True)
                objects_class[i] = classes[class_name]
            else:
                mask.append(False)
        objects_class = objects_class[mask]
        objects_pose = objects_pose[mask]
        objects_dimensions = objects_dimensions[mask]

        if len(objects_class) == 0:
            # If there are no labels, just create zeros.
            grid_x = math.floor(
                (x_max - x_min) / (step_x_size * downscaling_factor))
            grid_y = math.floor(
                (y_max - y_min) / (step_y_size * downscaling_factor))

            one_dim = np.zeros((grid_x, grid_y, anchor_dims.shape[0]),
                               dtype=np.float32)
            three_dim = np.zeros((grid_x, grid_y, anchor_dims.shape[0], 3),
                                 dtype=np.float32)
            class_dim = np.zeros(
                (grid_x, grid_y, anchor_dims.shape[0], len(classes)),
                dtype=np.float32)
            return one_dim, three_dim, three_dim, one_dim, one_dim, class_dim

        target_positions = objects_pose[:, 0:3]
        target_yaw = objects_pose[:, 3]
        target_dimension = objects_dimensions
        target_class = objects_class

        assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
        assert len(target_positions) == len(target_dimension) == len(
            target_yaw) == len(target_class)

        target = createPillarsTarget(target_positions, target_dimension,
                                     target_yaw, target_class, anchor_dims,
                                     anchor_z, anchor_yaw, pos_iou_threshold,
                                     neg_iou_threshold, angle_threshold,
                                     len(classes), downscaling_factor,
                                     step_x_size, step_y_size, x_min, x_max,
                                     y_min, y_max, z_min, z_max, False)
        target = target.astype(np.float32)

        def select(x, m):
            dims = np.indices(x.shape[1:])
            ind = (m,) + tuple(dims)
            return x[ind]

        best_anchors = target[..., 0:1].argmax(0)
        selection = select(target, best_anchors)

        # one hot encoding of class
        clf = selection[..., 9]
        clf[clf == -1] = 0
        ohe = np.eye(len(classes))[np.array(clf, dtype=np.int32).reshape(-1)]
        ohe = ohe.reshape(list(clf.shape) + [len(classes)])

        occupancy = selection[..., 0].astype(np.float32)
        position = selection[..., 1:4].astype(np.float32)
        size = selection[..., 4:7].astype(np.float32)
        angle = selection[..., 7].astype(np.float32)
        heading = selection[..., 8].astype(np.float32)
        classification = ohe.astype(np.float32)

        return occupancy, position, size, angle, heading, classification

    occupancy, position, size, angle, heading, classification = tf.numpy_function(
        make_ground_truth,
        inp=[objects_class, objects_pose, objects_dimensions],
        Tout=[
            tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
            tf.float32
        ],
        name="makeGroundTruth")
    occupancy.set_shape([None, None, None])
    position.set_shape([None, None, None, 3])
    size.set_shape([None, None, None, 3])
    angle.set_shape([None, None, None])
    heading.set_shape([None, None, None])
    classification.set_shape([None, None, None, None])

    return occupancy, position, size, angle, heading, classification


def drawOccupancyImage(occupancy: np.ndarray):
    """Draw RGB image for visualization of predicted occupancy for different anchors

    Args:
        occupancy (np.ndarray): predicted occupancy tensor of shape [x_size, y_size, num_anchors]

    Returns:
        np.ndarray: RGB image of shape [y_size, x_size, 3] visualizing occupancies for
            the first anchor in red, 2nd anchor in green and all other anchors in blue
    """
    # interpret first three anchor channels as RGB channels
    n_anchors = occupancy.shape[2]
    image = np.zeros((occupancy.shape[0], occupancy.shape[1], 3))
    if n_anchors <= 3:
        image[:, :, :n_anchors] = occupancy[:, :, :n_anchors]
    else:
        image[:, :, :2] = occupancy[:, :, :2]
        # merge anchors 3,4,... in blue channel
        image[:, :, 2] = np.max(occupancy[:, :, 2:], axis=-1)

    # order image dimensions
    image = np.flip(np.transpose(image, axes=(1, 0, 2)), axis=0)

    image = (255 * image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def boxesToCorners(x, y, yaw, length, width):

    sin = tf.math.sin(yaw)
    cos = tf.math.cos(yaw)
    half_length_times_cos = length * cos / 2
    half_length_times_sin = length * sin / 2
    half_width_times_cos = width * cos / 2
    half_width_times_sin = width * sin / 2

    x_minus_half_length_times_cos = x - half_length_times_cos
    y_minus_half_length_times_sin = y - half_length_times_sin

    x_plus_half_length_times_cos = x + half_length_times_cos
    y_plus_half_length_times_sin = y + half_length_times_sin

    x_1 = x_minus_half_length_times_cos + half_width_times_sin
    y_1 = y_minus_half_length_times_sin - half_width_times_cos
    x_2 = x_minus_half_length_times_cos - half_width_times_sin
    y_2 = y_minus_half_length_times_sin + half_width_times_cos
    x_3 = x_plus_half_length_times_cos - half_width_times_sin
    y_3 = y_plus_half_length_times_sin + half_width_times_cos
    x_4 = x_plus_half_length_times_cos + half_width_times_sin
    y_4 = y_plus_half_length_times_sin - half_width_times_cos

    return x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4


def drawObjectsOnImage(image: tf.Tensor,
                       object_list: tf.Tensor,
                       x_min: float,
                       x_max: float,
                       y_min: float,
                       y_max: float,
                       color: Tuple[int] = (255, 153, 255),
                       thickness: int = 1,
                       texts: Optional[str] = None,
                       font_scale: float = 0.4,
                       monte_carlo_dropout_use=False,
                       monte_carlo_dropout_ellipse=False,
                       monte_carlo_dropout_heading=False) -> tf.Tensor:
    """Draw 2D bounding boxes on bird's-eye-view image

    Args:
        image (tf.Tensor): image of shape (size_y, size_x, 3)
        object_list (tf.Tensor): objects (label, x, y, z, yaw, length, width, height) in list of shape [num_objects, 8]
        x_min (float): x coordinate of far left pixels
        x_max (float): x coordinate of far right pixels
        y_min (float): y coordinate of top pixels
        y_max (float): y coordinate of bottom pixels
        color (Tuple[int], optional): color of bounding box border line. Defaults to (255, 153, 255).
        thickness (int, optional): thickness of bounding box border line. Defaults to 2.
        texts [str, optional]: texts to be shown next to bounding boxes. Default to None.
        font_scale [float]: font scale for texts. Defaults to 0.4.
        monte_carlo_dropout_use (bool) : whether monte_carlo_dropout is on. Default to False.
        monte_carlo_dropout_ellipse (bool) : whether the uncertainty ellipse is visualized. Default to False.
        monte_carlo_dropout_heading (bool) : whether the uncertainty heading sector is visualized. Default to False.

    Returns:
        tf.Tensor: `image` with bounding boxes drawn on top
    """

    # pixels per unit
    px_per_unit_x = image.shape[1] / (x_max - x_min)
    px_per_unit_y = image.shape[0] / (y_max - y_min)

    # pixel coordinates of world origin
    origin_px_x = tf.cast((0 - x_min) * px_per_unit_x, dtype=tf.int32)
    origin_px_y = tf.cast((y_max - 0) * px_per_unit_y, dtype=tf.int32)

    if object_list is not None and tf.shape(object_list)[0] > 0:

        # extract object positions and dimensions
        x = object_list[..., I_X]
        y = object_list[..., I_Y]
        yaw = object_list[..., I_YAW]
        length = object_list[..., I_LENGTH]
        width = object_list[..., I_WIDTH]

        if monte_carlo_dropout_use:
            x_std = object_list[..., I_X_STD]
            y_std = object_list[..., I_Y_STD]
            x_ellip_leng = x_std * px_per_unit_x * 2 * np.sqrt(5.991)  # 95% confidence error ellipse. Find out how is np.sqrt(5.991) from in this article:
            y_ellip_leng = y_std * px_per_unit_y * 2 * np.sqrt(5.991)  # https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

        # get corners of boxes
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = boxesToCorners(
            x, y, yaw, length, width)

        # units to pixels
        x_1 = origin_px_x + tf.cast(x_1 * px_per_unit_x, tf.int32)
        y_1 = origin_px_y - tf.cast(y_1 * px_per_unit_y, tf.int32)
        x_2 = origin_px_x + tf.cast(x_2 * px_per_unit_x, tf.int32)
        y_2 = origin_px_y - tf.cast(y_2 * px_per_unit_y, tf.int32)
        x_3 = origin_px_x + tf.cast(x_3 * px_per_unit_x, tf.int32)
        y_3 = origin_px_y - tf.cast(y_3 * px_per_unit_y, tf.int32)
        x_4 = origin_px_x + tf.cast(x_4 * px_per_unit_x, tf.int32)
        y_4 = origin_px_y - tf.cast(y_4 * px_per_unit_y, tf.int32)
        x = origin_px_x + tf.cast(x * px_per_unit_x, tf.int32)
        y = origin_px_y - tf.cast(y * px_per_unit_y, tf.int32)

        # aggregate corner pixels
        point_1 = tf.stack([x_1, y_1], axis=-1)
        point_2 = tf.stack([x_2, y_2], axis=-1)
        point_3 = tf.stack([x_3, y_3], axis=-1)
        point_4 = tf.stack([x_4, y_4], axis=-1)
        points = tf.stack([point_1, point_2, point_3, point_4], axis=1)

        yaw_deg = -np.rad2deg(yaw)  # cast radian to angle

        # if the object list contains standard deviation
        if monte_carlo_dropout_use:
            for i in range(len(x)):
                if monte_carlo_dropout_ellipse:
                    ellipse_float = ((x[i],y[i]), (x_ellip_leng[i], y_ellip_leng[i]), 0.0)
                    image = tf.numpy_function(lambda image: cv2.ellipse(
                        image, ellipse_float, (255,255,255), 1),
                                              inp=[image],
                                              Tout=tf.float32)
                if monte_carlo_dropout_heading:
                    yaw_std = object_list[..., I_YAW_STD]
                    yaw_std = np.rad2deg(yaw_std)
                    # draw the uncertainty of the heading angle
                    if length[i] > 1.5:  # only draw heading uncertainty for those object with length > 5, that is, vehicle
                        image = tf.numpy_function(lambda image: cv2.ellipse(
                            image, (x[i], y[i]), (int(length[i]*3), int(width[i]*3)), int(yaw_deg[i]), int(2*yaw_std[i]),
                                                  int(-2*yaw_std[i]), (255, 255, 0), -1), inp=[image], Tout=tf.float32)

        # draw boxes on image
        image = tf.numpy_function(lambda image, points: cv2.drawContours(
            image, points, -1, color, thickness),
                                  inp=[image, points],
                                  Tout=tf.float32)
        if texts is not None:
            image = tf.map_fn(lambda elems: cv2.putText(
                elems[0].numpy(),
                str(elems[1].numpy())[:5], tuple(elems[2].numpy()), cv2.
                FONT_HERSHEY_DUPLEX, font_scale, color, 1),
                              elems=[
                                  tf.repeat(tf.expand_dims(image, axis=0),
                                            tf.shape(points)[0],
                                            axis=0), texts, point_1
                              ],
                              fn_output_signature=tf.float32)
            image = tf.reduce_max(image, axis=0)

    return image


def pointPillarsOutputToObjectList(output: tf.Tensor,
                                   anchors: List[List[float]],
                                   score_thresholds: List[float],
                                   x_min: float,
                                   y_min: float,
                                   step_x_size: float,
                                   step_y_size: float,
                                   downscaling_factor: int,
                                   max_width: float = 3.0,
                                   max_length: float = 30.0,
                                   max_height: float = 4.0,
                                   max_objects: int = 50,
                                   nms_iou_threshold: float = 0.1,
                                   monte_carlo_dropout_use: bool = False,
                                   is_label: bool = False
                                   ) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
    """Convert predicted tensors to object list

    Args:
        prediction (tf.Tensor): network
        anchors (List[List[float]]): list of anchors (length, width, height, z, yaw)
        score_thresholds (List[float]): List of minimum scores per class
        x_min (float): minimum x coordindate of points that are processed
        y_min (float): minimum y coordindate of points that are processed
        step_x_size (float): pillar size in x dimension
        step_y_size (float): pillar size in y dimension
        downscaling_factor (int): downscaling of pillar map to model outputs
        is_label (bool): Whether `output` is a label and not a prediction
        monte_carlo_dropout_use (bool) : whether monte_carlo_dropout is on. Default to False.


    Returns:
        Optional[tf.Tensor]: list of objects (class_id, x, y, z, yaw, length, width, height) in shape [num_objects, 8]
        Optional[tf.Tensor]: object scores in shape [num_objects]
    """

    if monte_carlo_dropout_use:
        occupancy, position, size, angle, heading, classification, pos_std = output
    else:
        occupancy, position, size, angle, heading, classification = output

    # occupancy: [height, width, num_anchors]
    # position: [height, width, num_anchors, 3]
    # size: [height, width, num_anchors, 3]
    # angle: [height, width, num_anchors]
    # heading: [height, width, num_anchors]
    # classification: [height, width, num_anchors, num_classes]

    # mean probability over all anchors
    mean_probability = tf.reduce_mean(occupancy, axis=-1)
    # mean_probability: [height, width]

    # one-hot-vector to class id
    classes = tf.math.argmax(classification, axis=-1)
    # classes: [height, width, num_anchors]

    # filter prediction by occupancy threshold
    thresholds = tf.gather(score_thresholds, classes)
    # thresholds: [height, width, num_anchors]

    if is_label:
        mask = occupancy > 0
    else:
        mask = occupancy >= thresholds
    # mask: [height, width, num_anchors]
    indices = tf.where(mask)
    i_x, i_y, i_anchor = tf.unstack(indices, axis=-1)
    # indices: [num_objects, 3] --> (i_x, i_y, i_anchor)

    occupancy = tf.boolean_mask(occupancy, mask)
    position = tf.boolean_mask(position, mask)
    size = tf.boolean_mask(size, mask)
    angle = tf.boolean_mask(angle, mask)
    heading = tf.boolean_mask(heading, mask)
    classification = tf.boolean_mask(classification, mask)
    if monte_carlo_dropout_use:
        pos_std = tf.boolean_mask(pos_std, mask)
    classes = tf.boolean_mask(classes, mask)
    cell_indices = tf.stack([i_x, i_y], axis=-1)
    mean_probability = tf.gather_nd(mean_probability, cell_indices)

    # return None if no box detected
    n = tf.shape(occupancy)[0]
    if n == 0:
        return None, None

    # get properties of selected anchors above threshold
    anchors_selected = tf.gather(anchors, i_anchor)
    anchor_dxyz = anchors_selected[:, 0:3]
    anchor_dd = tf.norm(anchor_dxyz[:, 0:2], axis=1)
    anchor_z = anchors_selected[:, 3]
    anchor_yaw = anchors_selected[:, 4]

    # compute cell centers
    x_cell = x_min + tf.cast(i_x, tf.float32) * step_x_size * downscaling_factor
    y_cell = y_min + tf.cast(i_y, tf.float32) * step_y_size * downscaling_factor
    # compute bounding boxes
    x = position[:, 0] * anchor_dd + x_cell
    y = position[:, 1] * anchor_dd + y_cell
    z = position[:, 2] * anchor_dxyz[:, 2] + anchor_z
    boxes_dimensions = tf.math.exp(size) * anchor_dxyz
    yaw = tf.math.asin(tf.clip_by_value(angle, -1.0, 1.0)) + anchor_yaw
    boxes_pose = tf.stack([x, y, z, yaw], axis=-1)
    boxes = tf.concat([boxes_pose, boxes_dimensions], axis=-1)

    scores = occupancy

    # filter boxes by nan values, heuristics, and lower probability than mean
    mask = tf.reduce_all(tf.math.is_finite(boxes), axis=-1)
    mask = tf.logical_and(
        tf.logical_and(tf.logical_and(mask, boxes[:, 4] <= max_length),
                       boxes[:, 5] <= max_width), boxes[:, 6] <= max_height)
    mask = tf.logical_and(mask, scores > mean_probability)

    # only keep relevant boxes
    classes = tf.boolean_mask(classes, mask)
    boxes = tf.boolean_mask(boxes, mask)
    scores = tf.boolean_mask(scores, mask)
    if monte_carlo_dropout_use:
        pos_std = tf.boolean_mask(pos_std, mask)

    if is_label:
        # assign score of 1.0 to label boxes
        scores = tf.ones_like(scores)

    # perform non maximum suppression to filter out duplicates from anchors
    if tf.shape(boxes)[0] > 0:
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = boxesToCorners(
            boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 5])
        boxes_corner = tf.stack([y_1, x_1, y_3, x_3], axis=-1)
        filtered_ind = tf.image.non_max_suppression(
            boxes_corner,
            scores,
            max_output_size=max_objects,
            iou_threshold=nms_iou_threshold)
        if monte_carlo_dropout_use:
            pos_std = tf.gather(pos_std, filtered_ind)
        boxes = tf.gather(boxes, filtered_ind)
        classes = tf.gather(classes, filtered_ind)
        scores = tf.gather(scores, filtered_ind)

    # return common object representation (class_id, x, y, z, yaw, length, width, height)
    if monte_carlo_dropout_use:
        object_list = tf.concat(
            [tf.expand_dims(tf.cast(classes, tf.float32), -1), boxes, pos_std[:, :2]], axis=-1)
    else:
        object_list = tf.concat(
            [tf.expand_dims(tf.cast(classes, tf.float32), -1), boxes], axis=-1)

    return object_list, scores


def countPointsIn3DBoundingBoxes(lidar_points: np.ndarray,
                                 objects_pose: np.ndarray,
                                 objects_dimensions: np.ndarray) -> np.ndarray:
    """Calculates the number of 3D points which lie inside of 3D bounding boxes

    Args:
        lidar_points (np.ndarray): array of lidar points containing x, y, z information has shape [num_points, 3]
        objects_pose (np.ndarray): array of object poses (x, y, z, yaw) in shape [num_objects, 4]
        objects_dimensions (np.ndarray): of object dimensions (length, width, height) in shape [num_objects, 3]

    Returns:
        np.ndarray: array of shape [num_objects, ] containing number of points that lie inside of the objects described in objects_pose and objects_dimensions
    """

    #  p8      p7
    #   +------+.
    #   |`.    | `.
    #   | P5+--+---+p6
    #   |   |  |   |             z
    # p4+---+--+p3 |         y   ^
    #    `. |    `.|          `. |
    #      `+------+            `+---> x
    #      p1       p2
    #
    # p1, p2, p5, p8 are used here

    # compute 4 adjacent corner points
    sin_yaw = np.sin(objects_pose[:, 3])
    cos_yaw = np.cos(objects_pose[:, 3])
    pose_x = objects_pose[:, 0]
    pose_y = objects_pose[:, 1]

    half_dimensions_x = objects_dimensions[:, 0] / 2
    half_dimensions_y = objects_dimensions[:, 1] / 2

    half_dimensions_x_times_cos_yaw = half_dimensions_x * cos_yaw
    half_dimensions_y_times_cos_yaw = half_dimensions_y * cos_yaw
    half_dimensions_x_times_sin_yaw = half_dimensions_x * sin_yaw
    half_dimensions_y_times_sin_yaw = half_dimensions_y * sin_yaw

    p1_x = pose_x - half_dimensions_x_times_cos_yaw + half_dimensions_y_times_sin_yaw
    p1_y = pose_y - half_dimensions_x_times_sin_yaw - half_dimensions_y_times_cos_yaw

    p2_x = pose_x + half_dimensions_x_times_cos_yaw + half_dimensions_y_times_sin_yaw
    p2_y = pose_y + half_dimensions_x_times_sin_yaw - half_dimensions_y_times_cos_yaw

    p8_x = pose_x - half_dimensions_x_times_cos_yaw - half_dimensions_y_times_sin_yaw
    p8_y = pose_y - half_dimensions_x_times_sin_yaw + half_dimensions_y_times_cos_yaw

    #p5_x = p1_x
    #p5_y = p1_y

    p1_z = objects_pose[:, 2] - objects_dimensions[:, 2] / 2
    # p2_z = p1_z
    # p8_z = p5_z
    p5_z = objects_pose[:, 2] + objects_dimensions[:, 2] / 2

    # compute dot product between difference and location vectors
    u_x = (p1_x - p2_x)
    u_y = (p1_y - p2_y)
    u_y_p1 = u_y * p1_y
    u_y_p2 = u_y * p2_y
    u_x_p1 = u_x * p1_x
    u_x_p2 = u_x * p2_x
    u_p1 = u_x_p1 + u_y_p1
    u_p2 = u_x_p2 + u_y_p2

    # compute dot product between difference and lidar points
    u_x_lp = np.tile(u_x, (lidar_points.shape[0], 1)).T * np.tile(
        lidar_points[:, 0], (u_y.shape[0], 1))
    u_y_lp = np.tile(u_y, (lidar_points.shape[0], 1)).T * np.tile(
        lidar_points[:, 1], (u_y.shape[0], 1))
    # dimensions: (#objects x #lidar_points)
    u_lp = u_x_lp + u_y_lp

    # compute dot product between difference and location vectors
    v_x = (p1_x - p8_x)
    v_y = (p1_y - p8_y)
    v_x_p1 = v_x * p1_x
    v_y_p1 = v_y * p1_y
    v_x_p8 = v_x * p8_x
    v_y_p8 = v_y * p8_y
    v_p1 = v_x_p1 + v_y_p1
    v_p8 = v_x_p8 + v_y_p8

    # compute dot product between difference and lidar points
    v_x_lp = np.tile(v_x, (lidar_points.shape[0], 1)).T * np.tile(
        lidar_points[:, 0], (v_x.shape[0], 1))
    v_y_lp = np.tile(v_y, (lidar_points.shape[0], 1)).T * np.tile(
        lidar_points[:, 1], (v_x.shape[0], 1))
    # dimensions: (#objects x #lidar_points)
    v_lp = v_x_lp + v_y_lp

    # compute dot product between difference and location vectors
    w_z = (p1_z - p5_z)
    w_p1 = w_z * p1_z
    w_p5 = w_z * p5_z
    # compute dot product between difference and lidar points
    w_lp = np.tile(w_z, (lidar_points.shape[0], 1)).T * np.tile(
        lidar_points[:, 2], (w_z.shape[0], 1))

    # check if dot product of lidar is between points
    bigger_x = v_lp <= np.tile(v_p1, (lidar_points.shape[0], 1)).T
    smaller_x = v_lp >= np.tile(v_p8, (lidar_points.shape[0], 1)).T
    inside_x = np.logical_and(bigger_x, smaller_x)

    # check if dot product of lidar is between points
    bigger_y = u_lp <= np.tile(u_p1, (lidar_points.shape[0], 1)).T
    smaller_y = u_lp >= np.tile(u_p2, (lidar_points.shape[0], 1)).T
    inside_y = np.logical_and(bigger_y, smaller_y)

    # check if dot product of lidar is between points
    bigger_z = w_lp <= np.tile(w_p1, (lidar_points.shape[0], 1)).T
    smaller_z = w_lp >= np.tile(w_p5, (lidar_points.shape[0], 1)).T
    inside_z = np.logical_and(bigger_z, smaller_z)

    # build complete mask
    inside = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)

    # dimensions: (#objects)
    sums_of_points_inside_bbox = np.sum(inside, 1)

    return sums_of_points_inside_bbox
