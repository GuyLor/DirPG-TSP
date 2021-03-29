import os
#import numpy as np
import itertools
#############   hyperparameters search    ##################


"""
# vars = [[attr1 val1 val2 val3], [attr2 val1 val2 val3], ...]
class Searcher:
    def __init__(self, args):
        base_command = ''
        for flag,val in args.base_command:
            base_command += "--{} {} ".format(flag, val)
        self.base_command = base_command

        self.num_exps = np.prod([len(var)-1 for var in args.variables])

        self.attrs_names = [var[0] for var in args.variables]
        self.attrs_vals = [var[1:] for var in args.variables]
        #self.experiments_flags = [name for name in self.attrs_names for vals in itertools.product(*self.attrs_vals) ]


    def experiments_generator(self):
        pass


    def run_experiment(self, command):
        print('#' * 150, '\n', command, '\n', '#' * 150)
        return os.system(command)

"""
def start(experiment):
    base_command = "python run.py" \
                   " --no_progress_bar" \
                   " --graph_size 20 " \
                   " --epoch_size 1280000" \
                   " --n_epochs 100 " \
                   " --exp_name supervised" \
                   " --total_interactions 5120000000" \
                   " --max_interactions 40 " \
                   " --annealing 0.0 "

    for exp in experiment(base_command):
        if run_experiment(exp) != 0:
            return

def run_experiment(command):
    print('#' * 150, '\n', command, '\n', '#' * 150)
    return os.system(command)

def alpha_experiment(base_command):
    alphas = [1.0, 5.0, 10.0, 3.0, 2.0]
    epsilons = [5.0, 10.0, 20.0, 2.0]
    dyn_weighting = [True, False]
    annealing = [0.0]
    for alp in alphas:
        for eps in epsilons:
            for d in dyn_weighting:
                for ann in annealing:
                    exp_command = " --alpha " + str(alp)
                    exp_command += " --epsilon " + str(eps)
                    exp_command += " --annealing " + str(ann)
                    if d:
                        exp_command += " --dynamic_weighting "

                    yield base_command+exp_command



def heuristic_experiment(base_command):
    heuristics = ['both'] #'mst', 'greedy',
    independent_gumbels = [False] #  [True, False]
    for ig in independent_gumbels:
        for h in heuristics:
            exp_command = " --heuristic " + str(h)
            if ig:
                exp_command += " --independent_gumbel "

            yield base_command + exp_command


def ablation_study(base_command):
    alphas = [0.0, 1.0, 1.2, 1.4, 1.6,1.8,2.0]
    epsilons = [0.5, 1.0, 2.0, 3.0]
    #dyn_weighting = [False]
    to_prune = [False] #, True]

    for alp in alphas:
        for eps in epsilons:
            for d in to_prune:
                exp_command = " --alpha " + str(alp)
                exp_command += " --epsilon " + str(eps)

                if d:
                    exp_command += " --not_prune "
                exp_command += " --run_name epsilon_{}_alpha_{}_dynamic_weighting_{}_not_prune_{}".format(eps, alp,False, d)
                yield base_command + exp_command


def gumbel_top_k(base_command):
    epsilons = [2.0, 3.0, 10.0, 20.0]
    #epsilons = [-1.0, -2.0, -3.0, -10]
    learning_rate = [1e-3, 1e-4, 1e-5]
    # to_prune = [True, False]
    np = False
    for lr in learning_rate:
        for eps in epsilons:
            exp_command = " --lr_model " + str(lr)
            exp_command += " --epsilon " + str(eps)
            #exp_command += " --alpha 1.6"
            if np:
                exp_command += " --not_prune "
            exp_command += " --run_name gumbel_top_k_epsilon_{}_lr_{}_not_prune_{}".format(eps, lr, np)
            yield base_command + exp_command

def reinforce(base_command):

    baselines = [None]  # 'rollout',
    bs = [20, 512]

    for bl in baselines:
        for b in bs:
            exp_command = " --no_dirpg "
            exp_command += " --batch_size "+ str(b)

            if bl is not None:
                exp_command += " --baseline " + str(bl)


            exp_command += " --run_name batch_size{}_baseline_{}".format(b, bl)
            yield base_command + exp_command

def dirpg_after_reinforce(base_command):

    alpha = 2.0
    epsilon = 5.0
    for epoch in range(10):
        exp_command = " --load_path" \
                      " outputs/tsp_20/aug28/REINFORCE_batch_size20_baseline_rollout_20200828T184532/epoch-{}.pt".format(epoch)
        exp_command += " --alpha " + str(alpha)
        exp_command += " --epsilon " + str(epsilon)
        exp_command += " --exp_name init_reinforce "
        exp_command += " --run_name start_from_epoch_{}".format(epoch)
        yield base_command + exp_command



def tsp_vs_mst(base_command):
    optimal = [True, False]
    alpha = [2.0,1.0,1.5]
    for alp in alpha:
        for opt in optimal:

            exp_command = " --not_prune "
            exp_command += " --optimal_heuristic "
            exp_command += " --alpha " + str(alp)
            exp_command += " --epsilon " + str(2)
            exp_command += " --run_name tsp_vs_mst_alpha_{}_optimal_{}".format(alp, opt)
            yield base_command + exp_command

def supervised(base_command):
    #epsilons = [-1.0, -2.0, -3.0, -10]
    learning_rate = [1e-4]
    # to_prune = [True, False]
    for lr in learning_rate:
        exp_command = " --lr_model " + str(lr)
        exp_command += " --supervised"
        exp_command += " --run_name supervised_concorde_lr_{}".format(lr)
        yield base_command + exp_command

start(supervised)


