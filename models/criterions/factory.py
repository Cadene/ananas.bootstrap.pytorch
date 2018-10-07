from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .multi_cross_entropy import MultiCrossEntropyLoss

def factory(engine=None, mode=None):

    if Options()['model']['criterion']['name'] == 'multi_cross_entropy':
        criterion = MultiCrossEntropyLoss()

    else:
        raise ValueError()

    return criterion