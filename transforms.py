import torch
import numpy as np
import random
import math

import scipy.spatial.distance
from torchvision import transforms

# from data_proc import read_off, path, pcshow, save_plotly_image


# Sample Points
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    # function to calculate triangle area by its vertices
    # Heron's formula: https://en.wikipedia.org/wiki/Heron%27s_formula
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    # barycentric coordinates on a triangle
    # barycentric coordinates: https://mathworld.wolfram.com/BarycentricCoordinates.html
    def sample_point(self, pt1, pt2, pt3):
        s,t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        # we sample 'k' faces with probabilities proportional to their areas
        # weights are used to create a distribution
        sampled_faces = (random.choices(faces, weights=areas, cum_weights=None, k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


# Normalize
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


# Augmentation
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

# rotate objects around Z-axis, so the rad with z-axis is constant
        theta = random.random() * 2. * math.pi  # rotation rad (in x-y plane)
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta),  math.cos(theta), 0],
                               [0,                0,               1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


# To Tensor
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

def train_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
])


# with open(path/"bed/train/bed_0001.off", 'r') as f:
#     verts, faces=read_off(f)

# pointcloud = PointSampler(3000)((verts, faces))
# norm_pointcloud = Normalize()(pointcloud)
# rot_pointcloud = RandRotation_z()(norm_pointcloud)
# noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)

# fig = pcshow(*noisy_rot_pointcloud.T)
# save_plotly_image(fig, 'transforms', 'jpg')

