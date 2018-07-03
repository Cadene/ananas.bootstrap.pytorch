import os
import click
import traceback
import collections
import queue
import threading
import torch
import argparse
from datetime import datetime
from bootstrap.run import init_experiment_directory
from bootstrap.lib import utils
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

# TODO
# Dev: Populate strategy
# Dev: History of mutation
# Dev: Save state / Resume state
# Dev: Mutation arch with pretraining
# Dev: Multiple metrics (min and max)
# Dev: Ranking strategies
# Dev: create views from papers
# Test: sgd pbt better than sgd hand tune
# Test: mutation arch converge
# Test: mutation arch pretraining converge

# SLURM SPECIFIC
import sys
sys.path.insert(0, '/home/rcadene/autopilotcv-pytorch')
import pyslurm
from slurm import GPU_QUEUE_MAP, submit, kill
from retrying import retry

# beware, retry hide errors
@retry
def get_job_status(job_id):
    job_info = pyslurm.job().get().get(job_id, {})
    return job_info.get('job_state', b'UNK')


def slurm_is_idle(job_id):
    return job_id is None or get_job_status(job_id) in [
        'BOOT_FAIL',
        'COMPLETED',
        'CANCELLED',
        'DEADLINE',
        'FAILED',
        'NODE_FAIL',
        'OUT_OF_MEMORY',
        'PREEMPTED',
        'RESV_DEL_HOLD',
        'REVOKED',
        'SPECIAL_EXIT',
        'STOPPED',
        'SUSPENDED',
        'TIMEOUT',
        'UNK',
    ]


def slurm_submit(cmd_args):
    slurm_args = argparse.Namespace(
        gpu_type='1080ti',
        num_gpus=1,
        cmd='python')
    job_id = submit(slurm_args, [cmd_args], verbose=False)
    return job_id


def find_ckpt(exp_dir, ckpt_name):
    path_ckpt = os.path.join(exp_dir, 'ckpt_{}_engine.pth.tar'.format(ckpt_name))
    if not os.path.isfile(path_ckpt):
        ckpt_name = 'last'
        path_ckpt = os.path.join(exp_dir, 'ckpt_{}_engine.pth.tar'.format(ckpt_name))
    return ckpt_name, path_ckpt


class Mutant():

    def __init__(self,
                 path_opts=None,
                 exp_dir=None,
                 generation=0,
                 status='spawned',
                 parent=None):
        self.path_opts = path_opts
        self.exp_dir = exp_dir
        self.generation = generation
        self.status = status
        self.parent = parent
        self.score = None
        #self.mutation = None

    def state_dict(self):
        state = {}
        state['path_opts'] = self.path_opts
        state['exp_dir'] = self.exp_dir
        state['generation'] = self.generation
        state['status'] = self.status
        state['parent'] = self.parent
        state['score'] = self.score
        return state

    def load_state_dict(self, state):
        for k,v in state.items():
            setattr(self, k, v)

    def train(self):
        self.status = 'training'
        opts = Options.load_yaml_opts(self.path_opts)
        Logger()('training {}'.format(self.path_opts))

        if self.generation == 0:
            ckpt_name = ''
        else:
            ckpt_name, _ = find_ckpt(self.exp_dir, Options()['pbt']['resume'])
        
        cmd = '{} -o {} --exp.dir {} --exp.resume {}  --engine.nb_epochs {}'.format(
            Options()['pbt']['cmd']['train'],
            self.path_opts, 
            self.exp_dir,
            ckpt_name,
            opts['engine']['nb_epochs'] + Options()['pbt']['mutation_rate'])
        Logger()(cmd)

        job_id = slurm_submit(cmd)
        while True:
            status = get_job_status(job_id)
            Logger()('training job {} is {} in {}'.format(job_id, status, self.exp_dir))
            if slurm_is_idle(job_id):
                break
            os.system('sleep 10')
        self.status = 'trained'

    def eval(self):
        self.status = 'evaluating'
        ckpt_name, path_ckpt = find_ckpt(self.exp_dir, Options()['pbt']['resume'])
        ckpt = torch.load(path_ckpt)

        metric_key = Options()['pbt']['resume'].replace('best_', '') # TODO
        self.score = ckpt['best_out'][metric_key]
        self.status = 'evaluated'
        # self.status = 'testing'
        # Logger()('\ntesting {}'.format(self.path_opts))
        # Logger()('{} -o {} --exp.dir {} --exp.resume {}'
        #       ' --dataset.train_split'
        #       ' --dateset.eval_split test'.format(
        #     Options()['pbt']['cmd']['test'],
        #     self.path_opts, 
        #     self.exp_dir,
        #     Options()['pbt']['resume']))
        # os.system('sleep 2')
        # self.status = 'tested'

    def clone(self):
        Logger()('cloning {}'.format(self.path_opts))
        exp_dir = os.path.join(
                os.path.dirname(self.exp_dir.strip()),
                '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
        path_opts = os.path.join(exp_dir, 'options.yaml')
        cmd = '{} {} {}'.format(
            Options()['pbt']['cmd']['clone'],
            self.exp_dir,
            exp_dir)
        Logger()(cmd)
        os.system(cmd)
        os.system('sleep 10')
        m = Mutant(path_opts, exp_dir,
            generation=self.generation+1,
            status='cloned',
            parent=self.path_opts)
        return m

    def evolve(self):
        self.status = 'evolving'
        Logger()('evolving {}'.format(self.path_opts))
        cmd = '{} --path_pbt_opts {} --path_mutant_opts {}'.format(
            Options()['pbt']['cmd']['evolve'],
            os.path.join(Options()['exp']['dir'], 'options.yaml'),
            self.path_opts
        )
        Logger()(cmd)
        job_id = slurm_submit(cmd)
        while True:
            status = get_job_status(job_id)
            Logger()('evolving job {} is {} in {}'.format(job_id, status, self.exp_dir))
            if slurm_is_idle(job_id):
                break
            os.system('sleep 10')
        self.status = 'evolved'


class Population():

    def __init__(self):
        self.queue = queue.Queue()
        self.pop = collections.OrderedDict()
        self.exp_dir = Options()['exp']['dir']
        self.path_ckpt = os.path.join(self.exp_dir, 'ckpt_population_last.pth.tar')
        self.n_pop_max = Options()['pbt']['n_pop_max']

    def populate(self):
        # TODO: improve populate
        path_mutant_opts = Options()['pbt']['path_opts']
        for i in range(Options()['pbt']['n_workers']):
            exp_mutant_dir = os.path.join(
                self.exp_dir,
                '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
            new_mutant = Mutant(path_mutant_opts, exp_mutant_dir)
            self.add_mutant_to_train(new_mutant)
            os.system('sleep 1')

    def add_mutant_to_train(self, mutant):
        if len(self.evaluated_mutants()) > self.n_pop_max:
            return
        self.pop[mutant.path_opts] = mutant
        self.queue.put(mutant)

    def get_mutant_to_train(self):
        m = self.queue.get()
        return m

    def resume(self):
        self.load_state_dict(torch.load(self.path_ckpt))
        # Resume training
        n_mutants = 0
        for k, m in self.pop.items():
            if m.status != 'killed':
                self.add_mutant_to_train(m)
                n_mutants += 1
                os.system('sleep 1')
        # TODO: multithread this
        # for i in range(n_mutants, Options['pbt']['n_workers']):
        #     best_mutant = self.best_mutant()
        #     new_mutant = best_mutant.clone()
        #     new_mutant.evolve()
        #     self.add_mutant_to_train(new_mutant)

    def save(self):
        torch.save(self.state_dict(), self.path_ckpt)

    def state_dict(self):
        state = {}
        state['pop'] = collections.OrderedDict()
        for k,m in self.pop.items():
            state['pop'][k] = m.state_dict()
        return state

    def load_state_dict(self, state):
        for k, mutant_state in state['pop'].items():
            m = Mutant()
            m.load_state_dict(mutant_state)
            self.pop[k] = m

    # def get_score(self, path_opts):
    #     return 1 # TODO:

    # def evaluate_population(self):
    #     for k, v in self.status.items():
    #         if k not in self.scores and v == 'killed':
    #             self.scores[k] = self.get_score(k)
    #     return 10 # TODO:


    def best_mutant(self):
        pop = {k:m for k, m in self.pop.items() if m.score is not None}
        ranked = sorted(self.pop.values(), key=lambda x:x.score, reverse=True) # TODO reverse=False
        # TODO not the best but random.choices 20% best
        return ranked[0]

    def evaluated_mutants(self):
        km = {k:v for k,v in self.pop.items() if v.score is not None}
        return km

    def __len__(self):
        return len(self.pop)

    def train_mutants(self):
        while True:
            mutant = self.get_mutant_to_train()
            if mutant is None:
                break

            if mutant.status != 'evaluated':
                mutant.train()
                self.save()

                mutant.eval()
                self.save()

            if len(self.evaluated_mutants()) > self.n_pop_max:
                self.queue.task_done()
                break

            best_mutant = self.best_mutant()
            new_mutant = best_mutant.clone()
            new_mutant.evolve()
            self.add_mutant_to_train(new_mutant)
            mutant.status = 'killed'
            self.save()
            self.queue.task_done()


def init_experiment_directory(exp_dir, resume=None):
    # create the experiment directory
    if not os.path.isdir(exp_dir):
        os.system('mkdir -p '+exp_dir)
    else:
        if resume is None:
            if click.confirm('Exp directory already exists in {}. Erase?'
                    .format(exp_dir, default=False)):
                os.system('rm -r '+exp_dir)
                os.system('mkdir -p '+exp_dir)
            else:
                os._exit(1)

    path_yaml = os.path.join(exp_dir, 'options.yaml')
    logs_name = 'logs'

    # create the options.yaml file
    Options().save(path_yaml)

    # open(write) the logs.txt file and init logs.json path for later
    Logger(exp_dir, name=logs_name)


def pbt(path_opts=None):
    global pop
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    Options(path_opts)

    # make exp directory if --exp.resume is empty
    init_experiment_directory(Options()['exp']['dir'], Options()['exp']['resume'])

    # initialiaze seeds to be able to reproduce experiment on reload
    utils.set_random_seed(Options()['misc']['seed'])

    # display and save options as yaml file in exp dir
    Logger().log_dict('options', Options(), should_print=True)

    pop = Population()

    threads = []
    for i in range(Options()['pbt']['n_workers']):
        t = threading.Thread(target=pop.train_mutants)
        t.start()
        threads.append(t)

    if Options()['exp']['resume'] is None:
        pop.populate()
    else:
        pop.resume()

    # block until all tasks are done
    pop.queue.join()

    # stop workers
    for i in range(Options()['pbt']['n_workers']):
        pop.queue.put(None)
    for t in threads:
        t.join()


def main(path_opts=None):
    try:
        pbt(path_opts=path_opts)
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

