# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SetIterInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        iter = runner.iter
        runner.model.roi_head.epoch = epoch
        runner.model.roi_head.iter = iter
        
