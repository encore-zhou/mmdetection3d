from mmcv.runner.hooks import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm


@HOOKS.register_module()
class BnUpdaterHook(Hook):
    """Batch Normalization Scheduler in MMDET3D.

    Args:
        by_epoch (bool): LR changes epoch by epoch
    """

    def __init__(self, step, gamma=0.1, by_epoch=True):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')

        self.by_epoch = by_epoch
        self.step = step
        self.gamma = gamma

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if runner.epoch not in self.step:
            return

        model = runner.model
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                prev_momentum = m.momentum
                update_momentum = prev_momentum * self.gamma
                m.momentum = update_momentum
