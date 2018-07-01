from bootstrap.lib.options import Options
from .cifar import CIFAR

def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'cifar10':    
        
        if Options()['dataset']['train_split']:
            dataset['train'] = factory_cifar10(Options()['dataset']['train_split'])

        if Options()['dataset']['eval_split']: 
            dataset['eval'] = factory_cifar10(Options()['dataset']['eval_split'])
    else:
        raise ValueError()

    return dataset


def factory_cifar10(split):
    dataset = CIFAR(
        Options()['dataset']['dir'],
        split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        shuffle=split=='train',
        name='CIFAR10')
    return dataset