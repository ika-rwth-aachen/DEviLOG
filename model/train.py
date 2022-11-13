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
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import random
import math

import utils
import point_cloud_utils
import object_utils
import grid_map_utils
import config
from metrics import EvidentialAccuracy
import tensorflow_datasets as tfds
import nuScenes


class LidarGridMapping():
    def __init__(self):
        conf = config.getConf()

        self.input_training = conf.input_training
        self.label_training = conf.label_training

        self.dataset = "nuscenes" if self.input_training == "nuscenes" else "files"

        self.batch_size = conf.batch_size
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

        self.nuscenes_raycasting = conf.nuscenes_raycasting
        self.nuscenes_sidewalk_is_occupied = conf.nuscenes_sidewalk_is_occupied
        self.nuscenes_min_points_in_bbox = conf.nuscenes_min_points_in_bbox
        self.point_distance_min = conf.point_distance_min

        self.label_has_dynamic_channel = True
        if self.label_has_dynamic_channel:
            self.channel_occ = 0
            self.channel_free = 1
            self.channel_occ_dyn = 2
        else:
            self.channel_occ = 1
            self.channel_free = 0
            self.channel_occ_dyn = None

        # load network architecture module
        architecture = utils.load_module(conf.model)

        if self.dataset == "nuscenes":
            # get max_samples_training random training samples
            nuscenes_train = tfds.load('nuscenes/lidar_cam_map', split='train')
            self.n_training_samples = len(nuscenes_train)
            print(f"Found {self.n_training_samples} training samples")

            # get max_samples_validation random validation samples
            nuscenes_valid = tfds.load('nuscenes/lidar_cam_map', split='valid')
            self.n_validation_samples = len(nuscenes_valid)
            print(f"Found {self.n_validation_samples} validation samples")

            # build samples with input = (point_cloud, image_front) and label = (objects, map)
            map_fn = lambda ex: (
                (ex["point_cloud"], ex["image_front"]), (ex["panoptic"], ex["objects"], ex["map"])
            )
            dataTrain = nuscenes_train.map(map_fn)
            dataValid = nuscenes_valid.map(map_fn)
        else:
            # get max_samples_training random training samples
            files_train_input = utils.get_files_in_folder(conf.input_training)
            files_train_label = utils.get_files_in_folder(conf.label_training)
            _, idcs = utils.sample_list(files_train_label,
                                        n_samples=conf.max_samples_training)
            files_train_input = np.take(files_train_input, idcs)
            files_train_label = np.take(files_train_label, idcs)
            self.n_training_samples = len(files_train_label)
            print(f"Found {self.n_training_samples} training samples")

            # get max_samples_validation random validation samples
            files_valid_input = utils.get_files_in_folder(conf.input_validation)
            files_valid_label = utils.get_files_in_folder(conf.label_validation)
            _, idcs = utils.sample_list(files_valid_label,
                                        n_samples=conf.max_samples_validation)
            files_valid_input = np.take(files_valid_input, idcs)
            files_valid_label = np.take(files_valid_label, idcs)
            self.n_validation_samples = len(files_valid_label)
            print(f"Found {self.n_validation_samples} validation samples")

            dataTrain = tf.data.Dataset.from_tensor_slices((files_train_input, files_train_label))
            dataValid = tf.data.Dataset.from_tensor_slices((files_valid_input, files_valid_label))

        # build training data pipeline
        dataTrain = dataTrain.shuffle(buffer_size=self.n_training_samples,
                                            reshuffle_each_iteration=True)
        # yapf: disable
        dataTrain = tf.data.Dataset.range(conf.epochs).flat_map(
            lambda e: tf.data.Dataset.zip((
                dataTrain,
                tf.data.Dataset.from_tensors(e).repeat(),
                tf.data.Dataset.range(self.n_training_samples)
            ))
        )
        dataTrain = dataTrain.map(lambda samples, *counters: samples + counters)
        # yapf: enable
        cardinality = conf.epochs * self.n_training_samples
        dataTrain = dataTrain.apply(
            tf.data.experimental.assert_cardinality(cardinality))
        dataTrain = dataTrain.map(
            self.parseSample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataTrain = dataTrain.batch(conf.batch_size, drop_remainder=True)
        dataTrain = dataTrain.repeat(conf.epochs)
        dataTrain = dataTrain.prefetch(1)
        print("Built data pipeline for training")

        # build validation data pipeline
        # yapf: disable
        dataValid = tf.data.Dataset.range(conf.epochs).flat_map(
            lambda e: tf.data.Dataset.zip((
                dataValid,
                tf.data.Dataset.from_tensors(e).repeat(),
                tf.data.Dataset.range(self.n_validation_samples)
            ))
        )
        dataValid = dataValid.map(lambda samples, *counters: samples + counters)
        # yapf: enable
        cardinality = conf.epochs * self.n_validation_samples
        dataValid = dataValid.apply(
            tf.data.experimental.assert_cardinality(cardinality))
        dataValid = dataValid.map(
            self.parseSample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataValid = dataValid.batch(conf.batch_size, drop_remainder=True)
        dataValid = dataValid.repeat(conf.epochs)
        dataValid = dataValid.prefetch(1)
        print("Built data pipeline for validation")

        # build model
        num_output_layers = 3 if self.label_has_dynamic_channel else 2
        model = architecture.getModel(
            self.y_min, self.y_max, self.x_min, self.x_max, self.step_x_size,
            self.step_y_size, self.max_points_per_pillar, self.max_pillars,
            self.number_features, self.number_channels,
            self.label_resize_shape, conf.batch_size, num_output_layers)
        if conf.model_weights is not None:
            model.load_weights(conf.model_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)
        loss = architecture.getLoss(self.label_has_dynamic_channel)
        metrics = [tf.keras.metrics.KLDivergence(), EvidentialAccuracy()]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Compiled model {os.path.basename(conf.model)}")
        model.summary()

        # create output directories
        model_output_dir = os.path.join(
            conf.output_dir,
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
        checkpoint_dir = os.path.join(model_output_dir, "Checkpoints")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # create callbacks to be called after each epoch
        tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_dir,
                                                        update_freq="epoch",
                                                        profile_batch=0)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "e{epoch:03d}_weights.hdf5"),
            period=conf.save_interval,
            save_weights_only=True)
        best_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_weights.hdf5"),
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True)

        callbacks = [
            tensorboard_cb, checkpoint_cb, best_checkpoint_cb
        ]

        # start training
        print("Starting training...")
        n_batches_train = self.n_training_samples // conf.batch_size
        n_batches_valid = self.n_validation_samples // conf.batch_size
        model.fit(dataTrain,
                  epochs=conf.epochs,
                  initial_epoch=conf.model_weights_epoch,
                  steps_per_epoch=n_batches_train,
                  validation_data=dataValid,
                  validation_freq=1,
                  validation_steps=n_batches_valid,
                  callbacks=callbacks)

    # build dataset pipeline parsing functions
    def parseSample(self, input_raw, label_raw=None, epoch=None, sample_idx=None):

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
