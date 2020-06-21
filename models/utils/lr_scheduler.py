import torch

class LRScheduler:
    def __init__(self, optimizer, init_lr: float = 3e-3, iters_per_step: int = 150000, decay_factor: float = 0.5):

        self.optimizer = optimizer

        self.curr_lr = init_lr

        self.curr_steps = 0
        self.iters_per_step = iters_per_step
        self.factor = decay_factor

    def step(self):
        if self.curr_steps >= self.iters_per_step:
            self.curr_lr = self.curr_lr * self.factor
            self.curr_steps = 0

        else:
            self.curr_steps += 1

        self.optimizer.param_groups[0]['lr'] = self.curr_lr
        return self.curr_lr

