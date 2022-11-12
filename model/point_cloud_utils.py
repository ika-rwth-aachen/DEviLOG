import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from point_pillars import createPillars
from pyntcloud import PyntCloud


def readPointCloud(
        file: Union[str, tf.Tensor],
        fields: List[str] = ["x", "y", "z", "intensity"]) -> np.ndarray:
    """Reads a point cloud from a PCD file.

    Args:
        file (Union[str, tf.Tensor]): path to pcd file
        fields (List[str]): fields to parse (["x", "y", "z", "intensity"])

    Returns:
        np.ndarray: list of points with selected fields in order
    """

    if tf.is_tensor(file):
        file = bytes.decode(file.numpy())

    # read fields
    point_cloud = PyntCloud.from_file(file).points
    point_cloud = point_cloud[fields].values

    # filter out nan values
    point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]

    return point_cloud


def normalizeIntensity(intensities, intensity_threshold=-1):
    if intensity_threshold == 0:
        intensities = tf.clip_by_value(intensities, 0.0, 0.0)
    elif intensity_threshold > 0:
        intensities = tf.clip_by_value(intensities / intensity_threshold, 0.0,
                                       1.0)

    return intensities


def makePointPillars(points: np.ndarray,
                     max_points_per_pillar,
                     max_pillars,
                     x_pixel_size,
                     y_pixel_size,
                     x_min,
                     x_max,
                     y_min,
                     y_max,
                     z_min,
                     z_max,
                     print_time=False,
                     min_distance=None):

    if min_distance is None:
        min_distance = -1

    pillars, indices = tf.numpy_function(func=createPillars,
                                         inp=[
                                             points, max_points_per_pillar,
                                             max_pillars, x_pixel_size,
                                             y_pixel_size, x_min, x_max, y_min,
                                             y_max, z_min, z_max, print_time,
                                             min_distance
                                         ],
                                         Tout=[tf.float32, tf.int32])

    # set Tensor shapes, as tf is unable to infer rank from tf.numpy_function
    pillars.set_shape([1, None, None, 9])
    indices.set_shape([1, None, 3])

    # remove batch dim from input tensors, will be added by data pipeline
    pillars = tf.squeeze(pillars, axis=0)
    indices = tf.squeeze(indices, axis=0)

    return pillars, indices


def lidarToBevImage(
    points_xy: Union[np.ndarray, tf.Tensor],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    x_pixel_size: float,
    y_pixel_size: float,
    points_intensity: Optional[Union[np.ndarray,
                                     tf.Tensor]] = None) -> np.ndarray:
    """Create bird's-eye-view image from lidar point cloud. Points are colored by their intensity.
       The x axis of the lidar coordinate system is oriented to the right and y upwards.

    Args:
        points_xy (Union[np.ndarray, tf.Tensor]): list of points (x, y).
        x_min (int): minimum x coordinate of points to be drawn.
        x_max (int): maximum x coordinate of points to be drawn.
        y_min (int): minimum y coordinate of points to be drawm.
        y_max (int): maximum y coordinate of points to be drawn.
        x_pixel_size (float): Size of one pixel in x dimension and unit of points coordinates.
        y_pixel_size (float): Size of one pixel in y dimension and unit of points coordinates.
        points_intensity (Union[np.ndarray, tf.Tensor]): list of point intensities in [0,1]. Defaults to None.

    Returns:
        np.ndarray: RGB image with shape [(y_max-y_min)/y_pixel_size, (x_max-x_min)/x_pixel_size, 3]
    """
    points_xy = tf.constant(points_xy)

    if len(points_xy.shape) != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected points to be of shape (num_points, 2) but has {points_xy.shape}.")
    if points_intensity is not None and (
            len(points_intensity.shape) != 1 or
            points_intensity.shape[0] != points_xy.shape[0]):
        raise ValueError(f"Expected intensities to be of shape (num_points) but has {points_intensity.shape}.")

    size_x = int((x_max - x_min) / x_pixel_size)
    size_y = int((y_max - y_min) / y_pixel_size)

    if points_intensity is None:
        points_intensity = [1.0] * len(points_xy)

    points_intensity = tf.constant(points_intensity)

    # deals with intensities out of bounds
    if tf.reduce_any(tf.logical_or(points_intensity > 1, points_intensity < 0)):
        minimum = tf.math.reduce_min(points_intensity)
        maximum = tf.math.reduce_max(points_intensity)
        if minimum < 0:
            print(
                f"Intensity {minimum} out of range [0,1]. Clipping visualization."
            )
        if maximum > 1:
            print(
                f"Intensity {maximum} out of range [0,1]. Clipping visualization."
            )

        tf.clip_by_value(points_intensity, 0., 1.)

    # scales points
    points_intensity *= 200
    points_intensity += 55

    # round values like opencv does before converting to uint8
    points_intensity = tf.math.round(points_intensity)

    # convert values to uint8
    points_intensity = tf.cast(points_intensity, tf.uint8)

    # adapt values
    points_xy -= [x_min, y_min]
    points_xy /= [x_pixel_size, y_pixel_size]

    # cast points to int
    points_xy = tf.cast(points_xy, tf.int16)

    points_xy = [0, size_y] - points_xy
    points_xy *= [-1, 1]

    # swap x and y, due to not using OpenCV anymore
    points_xy = tf.reverse(points_xy, [1])

    # use only points within image
    valid_x = tf.logical_and(points_xy[:, 0] >= 0, points_xy[:, 0] < size_y)
    valid_y = tf.logical_and(points_xy[:, 1] >= 0, points_xy[:, 1] < size_x)
    mask = tf.logical_and(valid_x, valid_y)
    points_xy = tf.boolean_mask(points_xy, mask)
    points_intensity = tf.boolean_mask(points_intensity, mask)

    # get the shape indices and updates for scatter_nd
    shape = tf.constant((size_y, size_x), tf.int32)
    indices = tf.cast(points_xy, tf.int32)
    updates = points_intensity

    # remove duplicates for the same cell as they would be accumulated by `tf.scatter_nd`
    indices, idxs = np.unique(indices, axis=0, return_index=True)
    updates = tf.gather(updates, idxs)

    # scatter points on image
    image = tf.scatter_nd(indices, updates, shape)

    # cast image back to numpy
    image = image.numpy().astype(np.uint8)

    # expand dimensions
    image = image[:, :, None]

    image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def rotatePointCloud(points: tf.Tensor, angle: float) -> tf.Tensor:
    """Rotate list of points by given angle in x-y-plane

    Args:
        points (tf.Tensor): list of points (x, y, z) in shape [num_points, 3]
        angle (float): rotation angle in rad

    Returns:
        tf.Tensor: list of rotated points in same shape as `points`
    """
    # yapf: disable
    rotation = tf.constant([[math.cos(angle), math.sin(angle), 0],
                            [-math.sin(angle), math.cos(angle), 0],
                            [0, 0, 1]])
    # yapf: enable
    points = tf.linalg.matmul(points, rotation)

    return points


def randomlyRemovePoints(
    lidar_points: np.ndarray,
    lidar_intensities: np.ndarray,
    random_point_removal_ratio: float,
    remaining_indices: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly removes points from a point cloud. The removed points are chosen randomly. 
    A ratio of random_point_removal_ratio of the original points are kept.
    The remaining_indices can optionally also be given from outside this function.


    Args:
        lidar_points (np.ndarray): Array of lidar points with intensities removed
        lidar_intensities (np.ndarray): Array of lidar intensities corresponding to the lidar_points
        random_point_removal_ratio (float): ratio [0, 1] of points that will not be discarded
        remaining_indices (np.ndarray): array of remaining indices of points in original point cloud

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of lidar points and intensities with 1-random_point_removal_ratio of the entries removed, array of remaining indices of points in original point cloud
    """

    if remaining_indices is None:
        original_number_of_points = lidar_points.shape[0]
        new_number_of_points = int(lidar_points.shape[0] *
                                   random_point_removal_ratio)
        remaining_indices = np.random.choice(original_number_of_points,
                                             new_number_of_points,
                                             replace=False)

    return lidar_points[remaining_indices], lidar_intensities[
        remaining_indices], remaining_indices
