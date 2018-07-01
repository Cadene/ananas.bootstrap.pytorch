import torch
import torch.optim.lr_scheduler
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger


def factory(model, engine=None):

    if Options()['optimizer']['name'] == 'cifar10':

        optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.network.parameters()),
                lr=Options()['optimizer']['lr'],
                momentum=Options()['optimizer']['momentum'],
                weight_decay=Options()['optimizer']['weight_decay'])

        if engine is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                Options()['engine']['nb_epochs'],
                last_epoch=engine.epoch-1)

            engine.register_hook('train_on_start_epoch', scheduler.step)
            engine.register_hook('train_on_start_epoch', log_lr(optimizer))
        
    return optimizer


def log_lr(optimizer):
    def func():
        Logger().log_value('train_epoch.lr',
            optimizer.param_groups[0]['lr'],
            should_print=False)
    return func
