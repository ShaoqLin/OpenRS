# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import math
import torch
import numpy as np
from fvcore.transforms.transform import (
    Transform,
    TransformList,
    NoOpTransform
)
from .transform import BoxNoiseTransform
import matplotlib.pyplot as plt

from detectron2.data.transforms.augmentation import Augmentation, _get_aug_input_args

__all__ = [
    "ImageAddNoise",
    "BoxAddNoise"
]

class ImageAddNoise(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        mutation_rate,
        _
    ):
        """
        Args:
            mutation_rate: the probability of mutation.
        """
        self.mutation_rate = mutation_rate
        self._init(locals())

    def get_transform(self, image):
        batch, w, h, c = image.size()
        mask_prob = torch.tensor([1.0-self.mutation_rate, self.mutation_rate])
        mask_indices = torch.multinomial(mask_prob.view(1, -1), w * h * c, replacement=True)
        mask = mask_indices.view(w, h, c).to(torch.float32)
        
        possible_mutations = torch.randint(0, 256, size = (w, h, c), dtype=torch.float32)
        x = torch.remainder((x.to(torch.int32) + mask*possible_mutations), 256)
        x = x.to(torch.float32)
             
        
class BoxAddNoise(Augmentation):
    def __init__(
        self,
        mutation_rate: float
    ):
        """
        Args:
            mutation_rate: the probability of mutation.        
        """
        self.mutation_rate = mutation_rate
        self._init(locals())

    def get_transform(self):
        return BoxNoiseTransform(mutation_rate=self.mutation_rate)
        
    def __call__(self, aug_input) -> Transform:
        """
        Augment the given `aug_input` **in-place**, and return the transform that's used.

        This method will be called to apply the augmentation. In most augmentation, it
        is enough to use the default implementation, which calls :meth:`get_transform`
        using the inputs. But a subclass can overwrite it to have more complicated logic.

        Args:
            aug_input (AugInput): an object that has attributes needed by this augmentation
                (defined by ``self.get_transform``). Its ``transform`` method will be called
                to in-place transform it.

        Returns:
            Transform: the transform that is applied on the input.
        """
        args = _get_aug_input_args(self, aug_input)
        
        # tfm = self.get_transform(*args, gt_box=gt_box)
        tfm = self.get_transform(*args)
        
        assert isinstance(tfm, (Transform, TransformList)), (
            f"{type(self)}.get_transform must return an instance of Transform! "
            "Got {type(tfm)} instead."
        )
        aug_input.transform(tfm)
        return tfm
