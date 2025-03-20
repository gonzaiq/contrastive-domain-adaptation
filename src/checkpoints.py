from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from typing import Optional

class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, min_steps_to_save: int = -1, max_steps_to_save: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_steps_to_save = min_steps_to_save  # start saving after this number of steps
        self.max_steps_to_save = max_steps_to_save  # stop saving after this number of steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if self.max_steps_to_save is not None and trainer.global_step > self.max_steps_to_save:
            return
        
        if trainer.global_step < self.min_steps_to_save:
            return
        
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
