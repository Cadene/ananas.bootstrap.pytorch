import os
import argparse
from bootstrap.lib import utils
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import init_experiment_directory
import random
import functools


def resume(path_opts=None, resume=None):
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    Options(path_opts)
    Options()['exp']['resume'] = resume

    # make exp directory if --exp.resume is empty
    init_experiment_directory(Options()['exp']['dir'], Options()['exp']['resume'])

    # initialiaze seeds to be able to reproduce experiment on reload
    utils.set_random_seed(Options()['misc']['seed'])

    # display and save options as yaml file in exp dir
    Logger().log_dict('options', Options(), should_print=True)

    # display server name and GPU(s) id(s)
    Logger()(os.uname())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        Logger()('CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

    # engine can train, eval, optimize the model
    # engine can save and load the model and optimizer
    engine = engines.factory()

    # dataset is a dictionary that contains all the needed datasets indexed by modes
    # (example: dataset.keys() -> ['train','eval'])
    engine.dataset = datasets.factory(engine)

    # model includes a network, a criterion and a metric
    # model can register engine hooks (begin epoch, end batch, end batch, etc.)
    # (example: "calculate mAP at the end of the evaluation epoch")
    # note: model can access to datasets using engine.dataset
    engine.model = models.factory(engine)

    # optimizer can register engine hooks
    engine.optimizer = optimizers.factory(engine.model, engine)

    # view will save a view.html in the experiment directory
    # with some nice plots and curves to monitor training
    engine.view = views.factory(engine)

    # load the model and optimizer from a checkpoint
    engine.resume()

    return engine
    

def process_mutations(d, out=None, key=None):
    if 'type' in d:
        d['key'] = key
        return {key: d}
    else:
        if out is None:
            out = {}
        for k,v in d.items():
            if key is None:
                new_key = k
            else:
                new_key = '{}.{}'.format(key,k)
            out.update(process_mutations(v, out, key=new_key))
        return out


# def rsetattr(obj, attr, val):
#     pre, _, post = attr.rpartition('.')
#     return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def apply_on_engine(mutation, engine):
    rgetattr(engine, mutation['key'])()


def apply_on_options(mutation, options):
    value = options[mutation['key']]
    if mutation['perturb_type'] == 'linear':
        new_v = value * random.choices(mutation['perturb'])
    elif mutation['perturb_type'] == 'logscale':
        new_v = -np.log10(1 - value)
        new_v = new_v * random.choices(mutation['perturb'])
        new_v = 1 - (1/(10 ** new_v))
    else:
        raise ValueError('Invalid perturb type found in config file.')
    options[mutation['key']] = new_value


def evolve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_pbt_opts', type=str, required=True)
    parser.add_argument('--path_mutant_opts', type=str, required=True)
    args = parser.parse_args()

    pbt_opts = Options.load_yaml_opts(args.path_pbt_opts)

    elements = []
    weights = []
    mutations = process_mutations(pbt_opts['pbt']['mutations'])
    keys = []
    weights = []
    for k, v in mutations.items():
        keys.append(k)
        weights.append(v['weight'])

    choices = random.choices(keys, weights, k=1)

    mutant_opts = Options.load_yaml_opts(args.path_mutant_opts)

    # WARNING: do not Logger()() anything before this step
    if sum(['method' in mutations[key] for key in choices]) > 0:
        engine = resume(args.path_mutant_opts, pbt_opts['pbt']['resume'])

    Logger()('mutation keys: {}'.format(keys))
    Logger()('mutation weights: {}'.format(weights))

    for key in choices:
        if mutations[key]['type'] == 'opt':
            apply_on_options(mutations[key], mutant_opts)
        elif mutations[key]['type'] == 'method':
            apply_on_engine(mutations[key], engine)
        else:
            raise ValueError()

    Options.save_yaml_opts(mutant_opts, args.path_mutant_opts)

    if sum(['method' in mutations[key] for key in choices]) > 0:
        engine.save(mutant_opts['exp']['dir'],
                    pbt_opts['pbt']['resume'],
                    engine.model,
                    engine.optimizer)


def main():
    try:
        evolve()
    # to avoid traceback for -h flag in arguments line
    except SystemExit:
        pass
    except:
        # to be able to write the error trace to exp_dir/logs.txt
        try:
            Logger()(traceback.format_exc(), Logger.ERROR)
        except:
            pass


if __name__ == '__main__':
    main()