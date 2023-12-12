import inspect


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def configs_as_dict(cfg):
    config = {}
    for i in inspect.getmembers(cfg):
        if not i[0].startswith('_'):
            if not inspect.ismethod(i[1]) and i[0] != 'torch':
                config[i[0]] = i[1]
    return config