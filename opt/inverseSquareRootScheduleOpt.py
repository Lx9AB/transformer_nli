

class InverseSquareRootScheduleOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, optimizer, lr, warmup_updates, warmup_init_lr=1e-6):
        self.step_num = 0
        self.lr = 0
        self.optimizer = optimizer
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.lr_step = (lr - warmup_init_lr) / warmup_updates
        self.decay_factor = lr * warmup_updates**0.5

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        """Update parameters and rate"""
        self.step_num += 1
        if self.step_num < self.warmup_updates:
            self.lr = self.warmup_init_lr + self.step_num*self.lr_step
        else:
            self.lr = self.decay_factor * self.step_num**-0.5
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
        self.optimizer.step()
