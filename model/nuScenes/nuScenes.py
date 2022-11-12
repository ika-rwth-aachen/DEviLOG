"""nuscenes dataset."""

import os
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds
from nuscenes import NuScenes
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap

_DESCRIPTION = """
The nuScenes dataset (pronounced /nuːsiːnz/) is a public large-scale dataset for autonomous driving developed by the team at Motional (formerly nuTonomy). Motional is making driverless vehicles a safe, reliable, and accessible reality. By releasing a subset of our data to the public, Motional aims to support public research into computer vision and autonomous driving.
"""

_CITATION = """
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and 
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and 
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
"""

VERSION = tfds.core.Version('1.3.0')
RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
    '1.1.0': 'Fixed class ids.',
    '1.2.0': 'Renamed to `nuscenes`. Added `image_front` feature.',
    '1.3.0': 'Add map feature. Add num lidar points to objects.'
}

MIN_POINTS_IN_BBOX = 5

def get_label_to_names_panoptic():
    return {
        0: 'noise',
        1: 'animal',
        2: 'human.pedestrian.adult',
        3: 'human.pedestrian.child',
        4: 'human.pedestrian.construction_worker',
        5: 'human.pedestrian.personal_mobility',
        6: 'human.pedestrian.police_officer',
        7: 'human.pedestrian.stroller',
        8: 'human.pedestrian.wheelchair',
        9: 'movable_object.barrier',
        10: 'movable_object.debris',
        11: 'movable_object.pushable_pullable',
        12: 'movable_object.trafficcone',
        13: 'static_object.bicycle_rack',
        14: 'vehicle.bicycle',
        15: 'vehicle.bus.bendy',
        16: 'vehicle.bus.rigid',
        17: 'vehicle.car',
        18: 'vehicle.construction',
        19: 'vehicle.emergency.ambulance',
        20: 'vehicle.emergency.police',
        21: 'vehicle.motorcycle',
        22: 'vehicle.trailer',
        23: 'vehicle.truck',
        24: 'flat.driveable_surface',
        25: 'flat.other',
        26: 'flat.sidewalk',
        27: 'flat.terrain',
        28: 'static.manmade',
        29: 'static.other',
        30: 'static.vegetation',
        31: 'vehicle.ego'
    }

def get_label_to_names_objects():
    return {
        0: 'Unclassified',
        1: 'Pedestrian',
        2: 'Bicycle',
        3: 'Motorbike',
        4: 'Car',
        5: 'Truck',
        6: 'Trailer',
        7: 'Bus',
        8: 'Animal',
        9: 'Road_Obstacle'
    }


class NuscenesConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Nuscenes."""

    def __init__(self, *, with_camera_images=False, with_map=False, **kwargs):
        """BuilderConfig for Nuscenes.
    Args:
      variant: str. Variant of the dataset.
      **kwargs: keyword arguments forwarded to super.
    """
        super(NuscenesConfig, self).__init__(version=VERSION, **kwargs)

        self.with_camera_images = with_camera_images
        self.with_map = with_map

class Nuscenes(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for nuscenes dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    You have to download files from https://www.nuscenes.org/nuscenes#download
    (This dataset requires registration).
    You have to download "Full Dataset", "Map Expansion Pack" and "nuScenes-panoptic".
    For testing purposes, you can just download the "mini" split of the full dataset.
    """

    BUILDER_CONFIGS = [
        NuscenesConfig(
            name="lidar",
            description="Full dataset with lidar, panoptic and objects",
        ),
        NuscenesConfig(
            name="lidar_map",
            description="Full dataset with lidar, panoptic, objects and map",
            with_map=True,
        ),
        NuscenesConfig(
            name="lidar_cam",
            description="Full dataset with lidar, panoptic, objects and camera images",
            with_camera_images=True,
        ),
        NuscenesConfig(
            name="lidar_cam_map",
            description="Full dataset with lidar, panoptic, objects, map and camera images",
            with_camera_images=True,
            with_map=True,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        if self._builder_config.with_camera_images and self._builder_config.with_map:
            features = tfds.features.FeaturesDict({
                    'point_cloud':  # list of points (x, y, z, intensity, ring)
                        tfds.features.Tensor(shape=(None, 5), dtype=tf.float32),
                    'panoptic':  # list of (class_id, instance_id) for each point in point_cloud
                        tfds.features.Tensor(shape=(None, 2), dtype=tf.uint16),
                    'objects':  # list of objects (class_id, x, y, z, yaw, length, width, height, num_lidar_points_in_bbox)
                        tfds.features.Tensor(shape=(None, 9), dtype=tf.float32),
                    "image_front":
                        tfds.features.Image(shape=(900, 1600, 3)),
                    "map":
                        tfds.features.Tensor(shape=(1000, 1000, 2), dtype=tf.uint8),
                })
        elif self._builder_config.with_camera_images:
            features = tfds.features.FeaturesDict({
                    'point_cloud':  # list of points (x, y, z, intensity, ring)
                        tfds.features.Tensor(shape=(None, 5), dtype=tf.float32),
                    'panoptic':  # list of (class_id, instance_id) for each point in point_cloud
                        tfds.features.Tensor(shape=(None, 2), dtype=tf.uint16),
                    'objects':  # list of objects (class_id, x, y, z, yaw, length, width, height, num_lidar_points_in_bbox)
                        tfds.features.Tensor(shape=(None, 9), dtype=tf.float32),
                    "image_front":
                        tfds.features.Image(shape=(900, 1600, 3)),
                })
        elif self._builder_config.with_map:
            features = tfds.features.FeaturesDict({
                    'point_cloud':  # list of points (x, y, z, intensity, ring)
                        tfds.features.Tensor(shape=(None, 5), dtype=tf.float32),
                    'panoptic':  # list of (class_id, instance_id) for each point in point_cloud
                        tfds.features.Tensor(shape=(None, 2), dtype=tf.uint16),
                    'objects':  # list of objects (class_id, x, y, z, yaw, length, width, height, num_lidar_points_in_bbox)
                        tfds.features.Tensor(shape=(None, 9), dtype=tf.float32),
                    "map":
                        tfds.features.Tensor(shape=(1000, 1000, 2), dtype=tf.uint8),
                })
        else:
            features = tfds.features.FeaturesDict({
                    'point_cloud':  # list of points (x, y, z, intensity, ring)
                        tfds.features.Tensor(shape=(None, 5), dtype=tf.float32),
                    'panoptic':  # list of (class_id, instance_id) for each point in point_cloud
                        tfds.features.Tensor(shape=(None, 2), dtype=tf.uint16),
                    'objects':  # list of objects (class_id, x, y, z, yaw, length, width, height, num_lidar_points_in_bbox)
                        tfds.features.Tensor(shape=(None, 9), dtype=tf.float32),
                })

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            supervised_keys=('point_cloud', 'objects'),
            homepage='https://www.nuscenes.org/nuscenes',
            citation=_CITATION,
            disable_shuffling=True
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path_full = dl_manager.manual_dir / "v1.0-trainval"
        path_mini = dl_manager.manual_dir / "v1.0-mini"
        if path_full.is_dir():
            path =  path_full
        elif path_mini.is_dir():
            path = path_mini
        else:
            raise ValueError("Could not find nuScenes folder 'v1.0-trainval' or 'v1.0-mini' in " + str(dl_manager.manual_dir))

        print("NuScenes root: " + str(path))

        if path == path_mini:
            self.nusc = NuScenes(version='v1.0-mini',
                                 dataroot=path,
                                 verbose=True)
            # nusc_mini.list_lidarseg_categories(sort_by='count', gt_from='panoptic')
        else:
            self.nusc = NuScenes(version='v1.0-trainval',
                                 dataroot=path,
                                 verbose=True)
            # nusc_trainval.list_lidarseg_categories(sort_by='count', gt_from='panoptic')

        if self._builder_config.with_map:
            self.nusc_map = {}
            self.nusc_map['boston-seaport'] = NuScenesMap(dataroot=path, map_name='boston-seaport')
            self.nusc_map['singapore-hollandvillage'] = NuScenesMap(dataroot=path, map_name='singapore-hollandvillage')
            self.nusc_map['singapore-onenorth'] = NuScenesMap(dataroot=path, map_name='singapore-onenorth')
            self.nusc_map['singapore-queenstown'] = NuScenesMap(dataroot=path, map_name='singapore-queenstown')

        return_dict = {
            'train': self._generate_examples(split='train'),
            'valid': self._generate_examples(split='val'),
        }

        return return_dict

    @staticmethod
    def get_class_id(class_name):
        if 'human.pedestrian' in class_name:
            class_id = 1
        elif 'vehicle.bicycle' in class_name:
            class_id = 2
        elif 'vehicle.motorcycle' in class_name:
            class_id = 3
        elif 'vehicle.car' in class_name or 'vehicle.emergency.police' in class_name:
            class_id = 4
        elif 'vehicle.truck' in class_name or 'vehicle.construction' in class_name or 'vehicle.emergency.ambulance' in class_name:
            class_id = 5
        elif 'vehicle.trailer' in class_name:
            class_id = 6
        elif 'vehicle.bus' in class_name:
            class_id = 7
        elif 'animal' in class_name:
            class_id = 8
        elif 'movable_object' in class_name or 'static_object' in class_name or 'static.' in class_name:
            class_id = 9
        elif 'flat.driveable_surface' in class_name:
            class_id = 10
        elif 'flat.' in class_name:
            class_id = 11
        elif 'vehicle.ego' in class_name:
            class_id = 12
        else:
            class_id = 0
        return class_id

    def _generate_examples(self, split=None):
        """Yields examples."""

        scene_splits = create_splits_scenes()
        i = 0
        for scene in self.nusc.scene:
            if scene['name'] in scene_splits[split]:
                sample_token = scene['first_sample_token']
                while sample_token != '':
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']

                    sample_data_lidar_top_token = sample['data']['LIDAR_TOP']
                    sample_data_lidar_top = self.nusc.get_sample_data(sample_data_lidar_top_token)

                    # point cloud
                    pcl_path = sample_data_lidar_top[0]
                    scan = np.fromfile(pcl_path, dtype=np.float32)
                    point_cloud = scan.reshape((-1, 5))

                    # panoptic segmentation
                    panoptic_path = os.path.join(
                        self.nusc.dataroot,
                        self.nusc.get('panoptic', sample_data_lidar_top_token)['filename'])
                    panoptic_raw = load_bin_file(panoptic_path, 'panoptic')
                    numpy_array_classes = np.array(panoptic_raw) // 1000
                    numpy_array_instances = np.mod(panoptic_raw, 1000)
                    panoptic = np.stack(
                        [numpy_array_classes, numpy_array_instances], axis=1)

                    # object list
                    object_list = []
                    for ann in sample_data_lidar_top[1]:
                        sample_annotation = self.nusc.get('sample_annotation', ann.token)
                        num_lidar_pts = sample_annotation['num_lidar_pts']
                        if num_lidar_pts > MIN_POINTS_IN_BBOX:
                            x = ann.center[0]
                            y = ann.center[1]
                            z = ann.center[2]
                            width = ann.wlh[0]
                            length = ann.wlh[1]
                            height = ann.wlh[2]
                            yaw = quaternion_yaw(ann.orientation)
                            class_id = self.get_class_id(ann.name)

                            object = (class_id, x, y, z, yaw, length, width, height, num_lidar_pts)
                            object_list.append(object)

                    if len(object_list) == 0:
                        object_list = np.empty((0, 9), dtype=np.float32)

                    # camera image
                    if self._builder_config.with_camera_images:
                        sample_data_cam_front_token = sample['data']['CAM_FRONT']
                        sample_data_cam_front = self.nusc.get_sample_data(sample_data_cam_front_token)
                        cam_front_image = sample_data_cam_front[0]

                    # map
                    if self._builder_config.with_map:
                        location = self.nusc.get('log', scene['log_token'])['location']
                        map = self.nusc_map[location]
                        # get ego pose
                        sample_data_lidar = self.nusc.get('sample_data', sample_data_lidar_top_token)
                        ego_pose = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
                        ego_translation = ego_pose['translation']
                        q = ego_pose['rotation']
                        ego_yaw = math.atan2(2.0*(q[1]*q[2] + q[3]*q[0]), q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2])
                        # get map
                        patch_box = (ego_translation[0], ego_translation[1], 200, 200)
                        patch_angle = -ego_yaw*180/math.pi
                        map_mask = map.get_map_mask(patch_box, patch_angle, ['drivable_area', 'walkway'], (1000, 1000))
                        map_mask = np.swapaxes(map_mask, 0, -1)

                    return_dict = {}
                    return_dict['point_cloud'] = point_cloud
                    return_dict['panoptic'] = panoptic
                    return_dict['objects'] = object_list
                    return_dict['panoptic'] = panoptic

                    if self._builder_config.with_camera_images:
                        return_dict['image_front'] = cam_front_image
                    if self._builder_config.with_map:
                        return_dict['map'] = map_mask

                    i += 1
                    yield i, return_dict
