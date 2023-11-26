# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import inspect
import pprint
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
import torch

from fvcore.transforms.transform import Transform, NoOpTransform, to_float_tensor, to_numpy

__all__ = [
    "BoxNoiseTransform",
]


class BoxNoiseTransform(Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, mutation_rate: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image.
                The two images must have the same shape, range, channel order
                and dtype.
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self.mutation_rate = mutation_rate
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the image.
        """
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is a no-op.
        """
        return NoOpTransform()

    def apply_noise(self, image: np.ndarray, boxes: str = None) -> np.ndarray:
        """
        Apply box noise on the image(s).
        """
        w, h, c = image.shape
        mask_prob = torch.tensor([1.0-self.mutation_rate, self.mutation_rate])
        mask_indices = torch.multinomial(mask_prob.view(1, -1), w * h * c, replacement=True).view(w, h, c).to(torch.float32)
        boxes_mask = torch.zeros(size=[w, h, c], dtype=torch.float32)
        for box in boxes:
            boxes_mask[int(box[1]): int(box[3]) + 1, int(box[0]): int(box[2]) + 1, :] = 1.0
        mask_indices = mask_indices * boxes_mask
        possible_mutations = torch.randint(0, 256, size = (w, h, c), dtype=torch.float32)
        image = torch.remainder((torch.tensor(image.copy()).to(torch.int32) + mask_indices*possible_mutations), 256)
        image = image.to(torch.float32)
        return image