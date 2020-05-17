import os

#############   hyperparameters search    ##################

def start(experiment):
    base_command = "python run.py" \
                   " --no_progress_bar" \
                   " --graph_size 20 " \
                   " --not_prune" \
                   " --exp_name heuristic_indep_gumbel" \
                   " --epoch_size 1000" \
                   " --n_epochs 20 " \
                   " --batch_size 1 " \
                   " --epsilon 20" \
                   " --annealing 0.0 " \
                   " --alpha 1" \
                   " --dynamic_weighting"

    experiment(base_command)



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

                    base_command += exp_command

    return base_command

def run_experiment(command):
    print('#' * 150, '\n', command, '\n', '#' * 150)
    try:
        os.system(command)
    except KeyboardInterrupt:
        return 'force stop'

def heuristic_experiment(base_command):
    heuristics = ['both'] #'mst', 'greedy',
    independent_gumbels = [False] #  [True, False]
    for ig in independent_gumbels:
        for h in heuristics:
            exp_command = " --heuristic " + str(h)
            if ig:
                exp_command += " --independent_gumbel "

            if run_experiment(base_command+exp_command) == 'force stop':
                return


start(heuristic_experiment)
