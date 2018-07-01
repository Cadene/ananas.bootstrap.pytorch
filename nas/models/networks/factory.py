import os
import torch.onnx
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .pnasnet import PNASNet
from .resnet import ResNet18
from .condense_net import condense_net
from .net_metrics import net_metrics
#from .utils import memory_usage

def factory(engine=None):

    Logger()('Creating cifar10 network...')

    if Options()['model']['network']['name'] == 'PNASNet':
        network = PNASNet(
            F=Options()['model']['network']['F'],
            n_heads=Options()['model']['network']['n_heads'])

    elif Options()['model']['network']['name'] == 'CondenseNet':
        network = condense_net()

    elif Options()['model']['network']['name'] == 'ResNet18':
        network = ResNet18()

    else:
        raise ValueError()

    if engine is not None:
        engine.register_hook('train_on_start', save_onnx(engine))
        # engine.register_hook('train_on_update', log_memory_used)
        # engine.register_hook('train_on_print', print_memory_used)
    
    return network


def save_onnx(engine):
    def func():
        path_onnx = os.path.join(Options()['exp']['dir'],'network.onnx')
        engine.model.eval()
        for batch in engine.dataset['eval'].make_batch_loader(batch_size=1):
            break
        batch = engine.model.prepare_batch(batch)

        Logger()('Exporting network to onnx')
        torch.onnx.export(engine.model.network, batch['data'], path_onnx, verbose=False)
        
        Logger()('Measuring network metrics')
        n_ops, n_params, fwd_dt = net_metrics(engine.model.network, data=batch['data'])
        Logger().log_value('net_metrics.n_ops', n_ops, should_print=False)
        Logger().log_value('net_metrics.n_params', n_params, should_print=False)
        Logger().log_value('net_metrics.fwd_dt', fwd_dt, should_print=False)
        Logger()('net_metrics.n_ops: {:.3f} millions of FLOPS'.format(n_ops/1000000))
        Logger()('net_metrics.n_params: {:.3f} millions of parameters'.format(n_params/1000000))
        Logger()('net_metrics.fwd_dt: {:.6f} seconds for a forward pass of 1 item'.format(fwd_dt))
    return func


# def log_memory_used():
#     item = memory_usage()
#     Logger().log_value('train_batch.memory.used', item['memory.used'], should_print=False)


# def print_memory_used():
#     Logger()('{} memory.used: {}'.format(' '*6, Logger().values['train_batch.memory.used'][-1]))
#     Logger()('{} memory.used: {:.5f}'.format(' '*6, Logger().values['train_batch.memory.used'][-1]))

