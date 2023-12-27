import torch
from detectron2.data.build import build_detection_train_loader
from detectron2.engine import HookBase
import detectron2.utils.comm as comm

class ValidationLossHook(HookBase):
    def __init__(self, cfg, DATASETS_VAL_NAME):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.defrost()
        self.cfg.DATASETS.TRAIN = DATASETS_VAL_NAME
        self.cfg.freeze()
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

