#!/usr/bin/env python

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

import os
import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import math

import utils
import point_cloud_utils
import object_utils
import grid_map_utils
import config
import tensorflow_datasets as tfds
import nuScenes

class LidarGridMapping():
    def __init__(self):
        conf = config.getConf()

        self.input_validation = conf.input_validation
        self.label_validation = conf.label_validation
        self.max_samples_testing = conf.max_samples_testing

        self.dataset = "nuscenes" if self.label_validation == "nuscenes" else "files"

        self.store_labels = conf.store_labels

        self.y_min = conf.y_min
        self.y_max = conf.y_max
        self.x_min = conf.x_min
        self.x_max = conf.x_max
        self.z_min = conf.z_min
        self.z_max = conf.z_max
        self.step_x_size = conf.step_x_size
        self.step_y_size = conf.step_y_size
        self.intensity_threshold = conf.intensity_threshold
        self.min_point_distance = conf.min_point_distance
        self.label_resize_shape = conf.label_resize_shape
        self.max_points_per_pillar = conf.max_points_per_pillar
        self.max_pillars = conf.max_pillars
        self.number_features = conf.number_features
        self.number_channels = conf.number_channels

        self.grid_config = grid_map_utils.GridConfig(x_min=conf.x_min, x_max=conf.x_max, y_min=conf.y_min, y_max=conf.y_max, z_min=conf.z_min, z_max=conf.z_max, 
            step_size_x=conf.step_x_size, step_size_y=conf.step_y_size)

        # TODO: config
        self.nuscenes_raycasting = False
        self.nuscenes_sidewalk_is_occupied = True
        self.nuscenes_min_points_in_bbox = 3
        self.point_distance_min = [3.0, 1.5]

        self.label_has_dynamic_channel = True
        if self.label_has_dynamic_channel:
            self.channel_occ = 0
            self.channel_free = 1
            self.channel_occ_dyn = 2
        else:
            self.channel_occ = 1
            self.channel_free = 0
            self.channel_occ_dyn = None
        
        self.model_weights = conf.model_weights

        # load network architecture module
        architecture = utils.load_module(conf.model)

        if self.dataset == "nuscenes":
            # get max_samples_validation random validation samples
            nuscenes_valid = tfds.load('nuscenes/lidar_cam_map', split='valid')
            n_samples = len(nuscenes_valid)
            print(f"Found {n_samples} validation samples")

            # build samples with input = (point_cloud, image_front) and label = (objects, map)
            map_fn = lambda ex: (
                (ex["point_cloud"], ex["image_front"]), (ex["panoptic"], ex["objects"], ex["map"])
            )
            dataValid = nuscenes_valid.map(map_fn)
        else:
            # get max_samples_validation random validation samples
            files_valid_input = utils.get_files_in_folder(self.input_validation)
            files_valid_label = utils.get_files_in_folder(self.label_validation)
            _, idcs = utils.sample_list(files_valid_label,
                                        n_samples=self.max_samples_testing)
            files_valid_input = np.take(files_valid_input, idcs)
            files_valid_label = np.take(files_valid_label, idcs)
            n_samples = len(files_valid_label)
            print(f"Found {n_samples} validation samples")

            dataValid = tf.data.Dataset.from_tensor_slices((files_valid_input, files_valid_label))

        dataValid = dataValid.map(self.parseSample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        num_output_layers = 3 if self.label_has_dynamic_channel else 2
        model = architecture.getModel(self.y_min, self.y_max, self.x_min, self.x_max,
                                    self.step_x_size, self.step_y_size,
                                    self.max_points_per_pillar, self.max_pillars,
                                    self.number_features, self.number_channels,
                                    self.label_resize_shape, 1, num_output_layers)
        model.load_weights(self.model_weights)
        print(f"Reloaded model from {self.model_weights}")

        # evaluate
        print("Evaluating ...")
        eval_dir = os.path.join(os.path.dirname(self.model_weights), os.pardir,
                                "Evaluation")

        # evaluation metrics
        evaluation_dict = {}
        evaluation_dict['deep'] = {}
        evaluation_dict['deep']['KL_distance'] = []
        evaluation_dict['deep']['m_unknown'] = []
        evaluation_dict['deep']['m_occupied'] = []
        evaluation_dict['deep']['m_free'] = []
        evaluation_dict['naive'] = {}
        evaluation_dict['naive']['KL_distance'] = []
        evaluation_dict['naive']['m_unknown'] = []
        evaluation_dict['naive']['m_occupied'] = []
        evaluation_dict['naive']['m_free'] = []

        i = 0
        for sample in tqdm.tqdm(dataValid):
            input, label = sample
            pillars = tf.expand_dims(input[0], 0)
            voxels = tf.expand_dims(input[1], 0)
            prediction = model.predict((pillars, voxels)).squeeze()

            i += 1
            sample_name = str(i)

            kld = tf.keras.metrics.KLDivergence()

            # collect belief masses and Kullback-Leibler distance for predictions by deep ISM
            prob, u, _, _ = utils.evidences_to_masses(prediction)
            evaluation_dict['deep']['m_unknown'].append(float(tf.reduce_mean(u)))
            evaluation_dict['deep']['m_free'].append(
                float(tf.reduce_mean(prob[..., 0])))
            evaluation_dict['deep']['m_occupied'].append(
                float(tf.reduce_mean(prob[..., 1])))
            evaluation_dict['deep']['KL_distance'].append(float(kld(label,
                                                                    prediction)))

            # save predicted grid map
            prediction_dir = os.path.join(eval_dir, "predictions")
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            prediction_img = utils.evidence_to_ogm(prediction)
            output_file = os.path.join(prediction_dir,
                                    sample_name)
            cv2.imwrite(output_file + ".png",
                        cv2.cvtColor(prediction_img, cv2.COLOR_RGB2BGR))

            # save label as image
            if self.store_labels:
                label_img = utils.evidence_to_ogm(label)
                label_dir = os.path.join(eval_dir, "labels")
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                output_file = os.path.join(label_dir, sample_name)
                cv2.imwrite(output_file + ".png", cv2.cvtColor(label_img,
                                                            cv2.COLOR_RGB2BGR))

        # create subfolders
        plot_dir = os.path.join(eval_dir, "plots")
        raw_dir = os.path.join(eval_dir, "raw")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        # plot cross entropy over evaluation dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time in seconds')

        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w',
                    top=False,
                    bottom=False,
                    left=False,
                    right=False)

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        t = np.arange(0, len(evaluation_dict['naive']['m_unknown']))
        ax1.plot(t, evaluation_dict['deep']['m_unknown'], 'b-', t,
                evaluation_dict['deep']['m_free'], 'g-', t,
                evaluation_dict['deep']['m_occupied'], 'r-', t,
                evaluation_dict['naive']['m_unknown'], 'b--', t,
                evaluation_dict['naive']['m_free'], 'g--', t,
                evaluation_dict['naive']['m_occupied'], 'r--')
        ax1.set_ylim(0, 1.0)
        ax1.legend([
            r'$\overline{m}(\Theta)$', r'$\overline{m}(F)$', r'$\overline{m}(O)$',
            r'$\overline{m}_G(\Theta)$', r'$\overline{m}_G(F)$', r'$\overline{m}_G(O)$'
        ])

        ax2.plot(t, evaluation_dict['deep']['KL_distance'], 'k-', t,
                evaluation_dict['naive']['KL_distance'], 'k--')
        ax2.legend([
            r'$KL\left[Dir(p|\hat{\alpha})||Dir(p|\alpha)\right]$',
            r'$KL\left[Dir(p|\hat{\alpha}_G)||Dir(p|\alpha)\right]$'
        ])

        plt.savefig(os.path.join(plot_dir, 'evaluation.png'))

        # store values as json file
        evaluation_json = dict()
        evaluation_json['eval_kld'] = np.vstack(
            (t, evaluation_dict['deep']['KL_distance'])).transpose().tolist()
        evaluation_json['eval_uncertainty'] = np.vstack(
            (t, evaluation_dict['deep']['m_unknown'])).transpose().tolist()
        evaluation_json['eval_prob_free'] = np.vstack(
            (t, evaluation_dict['deep']['m_free'])).transpose().tolist()
        evaluation_json['eval_prob_occupied'] = np.vstack(
            (t, evaluation_dict['deep']['m_occupied'])).transpose().tolist()

        evaluation_json['eval_naive_kld'] = np.vstack(
            (t, evaluation_dict['naive']['KL_distance'])).transpose().tolist()
        evaluation_json['eval_naive_uncertainty'] = np.vstack(
            (t, evaluation_dict['naive']['m_unknown'])).transpose().tolist()
        evaluation_json['eval_naive_prob_free'] = np.vstack(
            (t, evaluation_dict['naive']['m_free'])).transpose().tolist()
        evaluation_json['eval_naive_prob_occupied'] = np.vstack(
            (t, evaluation_dict['naive']['m_occupied'])).transpose().tolist()
        with open(os.path.join(raw_dir, 'evaluation.json'), 'w') as fp:
            json.dump(evaluation_json, fp)

    # build dataset pipeline parsing functions
    def parseSample(self, input_raw, label_raw):

        if self.dataset == "nuscenes":
            # generate samples from nuScenes dataset
            if type(input_raw) is tuple:
                input_raw = input_raw[0]
            lidar_points = input_raw[:, 0:3]
            lidar_intensities = input_raw[:, 3:4]
            if label_raw is None:
                grid_map = None
            else:
                label_panoptic, label_objects, label_map = label_raw

                # rotate point cloud and object list so that driving direction points upwards
                angle = -1.0 * math.pi / 2.0
                lidar_points = point_cloud_utils.rotatePointCloud(lidar_points, angle)
                objects_class, objects_pose, objects_dimensions = object_utils.readObjectList(label_objects, label_type='nuscenes')
                objects_pose = object_utils.rotateObjectList(objects_pose, angle)
                objects_lidar_pts = label_objects[..., 8:9]
                object_list = tf.concat([objects_pose, objects_dimensions, tf.expand_dims(tf.cast(objects_class, tf.float32), -1), objects_lidar_pts], axis=-1)

                # create "label" grid map from nuscenes map and object list
                if self.nuscenes_sidewalk_is_occupied:
                    map_drivable_space = tf.cast(tf.reduce_any(label_map > 0, axis=-1), tf.uint8)
                else:
                    map_drivable_space = label_map[..., 0]
                grid_map = grid_map_utils.mapToOgm(map_drivable_space, object_list, self.grid_config, not self.label_has_dynamic_channel, self.nuscenes_raycasting, self.nuscenes_min_points_in_bbox, self.point_distance_min)
        else:
            # generate samples from file
            lidar = tf.py_function(func=point_cloud_utils.readPointCloud, inp=[input_raw], Tout=tf.float32)
            lidar_points = lidar[..., 0:3]
            lidar_intensities = lidar[..., 3:4]

            if label_raw is None:
                grid_map = None
            else:
                # convert PNG file (grid map) to matrix
                grid_map = tf.image.decode_png(tf.io.read_file(label_raw))

                if self.label_has_dynamic_channel:
                    # use channels 'm_occupied_dynamic', 'm_free' and 'm_occupied_static'
                    grid_map = tf.cast(grid_map[..., 0:3], tf.float32)
                else:
                    # use channels 'm_free' (green) and 'm_occupied' (red)
                    grid_map = tf.cast(grid_map[..., 1:3], tf.float32)

                # normalize from image [0..255] to [0.0..1.0]
                grid_map = tf.divide(grid_map, 255.0)

        # normalize intensities in point cloud
        lidar_intensities = point_cloud_utils.normalizeIntensity(
            lidar_intensities, self.intensity_threshold)
        
        if grid_map is not None:
            # restore shape information
            if self.label_has_dynamic_channel:
                grid_map.set_shape([None, None, 3])
            else:
                grid_map.set_shape([None, None, 2])

            # resize grid map to model output size
            grid_map = tf.image.resize(
                grid_map,
                self.label_resize_shape[0:2],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # create model input
        lidar = tf.concat([lidar_points, lidar_intensities], -1)
        pillars, voxels = utils.make_point_pillars(
            lidar, self.max_points_per_pillar, self.max_pillars,
            self.step_x_size, self.step_y_size, self.x_min, self.x_max,
            self.y_min, self.y_max, self.z_min, self.z_max,
            min_distance = self.min_point_distance)

        network_inputs = (pillars, voxels)
        if label_raw is not None:
            network_labels = (grid_map)
        else:
            network_labels = None

        return network_inputs, network_labels

if __name__ == '__main__':
    i = LidarGridMapping()
