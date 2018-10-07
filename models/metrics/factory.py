from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .multi_accuracy import MultiAccuracy

def factory(engine=None, mode=None):

    if Options()['model']['metric']['name'] == 'multi_accuracy':
        metric = MultiAccuracy(topk=Options()['model']['metric']['topk'])

    else:
        raise ValueError()

    return metric
