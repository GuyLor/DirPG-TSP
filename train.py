import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values, log_values_dirpg, log_values_supervised
from utils import move_to

from datetime import datetime, timedelta

from contextlib import contextmanager
import sys, os
import contextlib

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
global_avg_reward = 0


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    model = model.model if not opts.no_dirpg else model
    get_inner_model(model).decoder.interactions_count = False
    model.eval()
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    #get_inner_model(model).decoder.interactions_count = True
    model.train()

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!

    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), only_encoder=False)
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch,interactions_count, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard and opts.no_dirpg:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    if not opts.no_dirpg:
        tb_logger.add_scalar('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    get_inner_model(model).decoder.count_interactions = True
    if baseline.__class__.__name__ != "NoBaseline":
        get_inner_model(baseline.baseline.model).decoder.count_interactions = True
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution, circles=False))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    global global_avg_reward
    # Put model in train mode!
    if opts.no_dirpg:
        model.train() if opts.no_dirpg else model.model.train()
        set_decode_type(model, "sampling")
    else:
        model.model.train()

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        if opts.no_dirpg:
            train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                interactions_count,
                batch,
                tb_logger,
                opts
            )

        else:
            try:
                train_dirpg_batch(
                    model,
                    optimizer,
                    epoch,
                    batch_id,
                    step,
                    interactions_count,
                    batch,
                    tb_logger,
                    opts,

                )
            except KeyboardInterrupt:
                tb_logger.add_hparams({'batch size': opts.batch_size,
                                       'epsilon': opts.epsilon,
                                       'alpha': opts.alpha,
                                       'lr': opts.lr_model,
                                       'heuristic': opts.heuristic,
                                       'eps anneal factor': opts.annealing,
                                       'dynamic weighting': opts.dynamic_weighting,
                                       'max_interactions': opts.max_interactions,
                                       'not_prune': opts.not_prune,
                                       'k_improvement': opts.k_improvement},
                                      {'cost': global_avg_reward})

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model if opts.no_dirpg else model.model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    get_inner_model(model).decoder.count_interactions = False
    if baseline.__class__.__name__ != "NoBaseline":
        get_inner_model(baseline.baseline.model).decoder.count_interactions = False

    avg_reward = validate(model, val_dataset, opts)
    global_avg_reward = avg_reward

    if not opts.no_tensorboard and opts.no_dirpg:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
    elif not opts.no_dirpg:
        tb_logger.add_scalar('val_avg_reward', avg_reward, step)

    if not opts.no_dirpg and epoch == opts.n_epochs - 1:
        tb_logger.add_hparams({'batch size': opts.batch_size,
                               'epsilon': opts.epsilon,
                               'alpha': opts.alpha,
                               'lr': opts.lr_model,
                               'heuristic': opts.heuristic,
                               'eps annealing factor': opts.annealing,
                               'dynamic weighting': opts.dynamic_weighting,
                               'max_interactions': opts.max_interactions,
                               'not_prune': opts.not_prune,
                               'k_improvement': opts.k_improvement},
                              {'cost': global_avg_reward})
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def GetTime(sec):
    sec = timedelta(seconds=int(sec))
    d = datetime(1,1,1) + sec

    print("ETA (D:H:M)", end=" ")
    print("%d:%d:%d" % (d.day-1, d.hour, d.minute))


sum_batch_time = 0
def train_dirpg_batch(
                dirpg_trainer,
                optimizer,
                epoch,
                batch_id,
                step,
                interactions_count,
                batch,
                tb_logger,
                opts
):

    start_time = time.time()

    x = move_to(batch, opts.device)
    # Evaluate model, get costs and log probabilities
    # dirpg_trainer.search_params['alpha'] = np.min([opts.alpha*math.exp(0.002 * step), 4.0])
    alp = np.max([opts.alpha * math.exp(-opts.annealing * step), 1.3])
    #eps = np.max([opts.epsilon*math.exp(-opts.annealing * step), opts.min_eps]) #
    #eps = opts.epsilon  #  if step % 2 == 0 else -opts.epsilon
    eps = 30*torch.rand(1).item()
    budget = opts.max_interactions # 64
    """
    if step > 200 or step == 0: # and step > 0:
        #opts.k_improvement = 1
        #budget = opts.max_interactions
        #eps = 0.6
        #opts.alpha = 1.6
        dirpg_trainer.a_star_cpp.setToPrint(True)
    else:
        dirpg_trainer.a_star_cpp.setToPrint(False)
    """
    if not opts.supervised:
        dirpg_trainer.a_star_cpp.setEpsilonAlpha(eps, opts.alpha)
        dirpg_trainer.a_star_cpp.setKImprovement(opts.k_improvement)
        direct_loss, to_log = dirpg_trainer.train_dirpg(x, budget=budget, epsilon=eps)
    else:

        direct_loss, to_log = dirpg_trainer.train_with_concorde(x)
    if direct_loss is not None:
        loss = direct_loss.sum()
        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()
    else:
        grad_norms = [0], [0]

    # Logging
    global sum_batch_time
    sum_batch_time += (time.time() - start_time)
    #print("interactions_count: ", interactions_count)
    #print("dirpg_trainer.decoder.interactions: ")
    #print(dirpg_trainer.decoder.interactions)
    to_log['interactions'] = interactions_count + dirpg_trainer.decoder.interactions
    if step % int(opts.log_step) == 0:
        if not opts.supervised:
            log_values_dirpg(to_log, grad_norms, epoch, batch_id, step, tb_logger, opts)
            eps, alpha = dirpg_trainer.a_star_cpp.getEpsilonAlpha()
            print("epsilon: {}, alpha: {}".format(eps, alpha))
        else:
            log_values_supervised(to_log, grad_norms, epoch, batch_id, step, tb_logger, opts)

        interactions_so_far = to_log["interactions"]
        #print(opts.total_interactions)
        #print(interactions_so_far)
        if opts.total_interactions > interactions_so_far:
            avg_batch_time = (sum_batch_time/interactions_so_far).item() if step > 0 else sum_batch_time
            print('batch time: ',avg_batch_time)
            #seconds = (opts.n_epochs*(opts.epoch_size // opts.batch_size) - step) * avg_batch_time
            seconds = (opts.total_interactions - interactions_so_far) * avg_batch_time
            GetTime(seconds)
            #opts.epsilon = -opts.epsilon
            #dirpg_trainer.a_star_cpp.setEpsilonAlpha(np.random.uniform(0.01, 5.0, 1).item(), np.random.uniform(1.3, 3, 1).item())
        print('============================')

    if opts.use_cuda:
        torch.cuda.empty_cache()




def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        interactions_count,
        batch,
        tb_logger,
        opts
):
    start_time = time.time()
    get_inner_model(model).decoder.count_interactions = True
    get_inner_model(baseline.baseline.model).decoder.count_interactions = True
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities

    cost, log_likelihood = model(x, only_encoder=False)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    global sum_batch_time
    sum_batch_time += (time.time() - start_time)
    # Logging
    interactions_so_far = interactions_count +\
                          2*get_inner_model(model).decoder.interactions*torch.cuda.device_count()

    if step % int(opts.log_step) == 0:
        #with torch.no_grad():
        #    opt_nll = get_inner_model(model).supervised_log_likelihood(x)
        opt_nll=torch.ones(1)
        log_values(cost, grad_norms, epoch, interactions_so_far, batch_id, step,
                   log_likelihood, opt_nll, reinforce_loss, bl_loss, tb_logger, opts)

        #interactions_so_far = 2 * step * opts.graph_size * opts.batch_size \
        #    if opts.baseline is not None else step * opts.graph_size * opts.batch_size



        if opts.total_interactions > interactions_so_far:
            avg_batch_time = sum_batch_time/step if step > 0 else sum_batch_time
            print('batch time: ',avg_batch_time)

            seconds = (opts.total_interactions - interactions_so_far) * avg_batch_time
            #seconds = (opts.n_epochs*(opts.epoch_size // opts.batch_size) - step) * avg_batch_time
            GetTime(seconds)
            print('============================')

    get_inner_model(model).decoder.count_interactions = False
    get_inner_model(baseline.baseline.model).decoder.count_interactions = False

    if opts.use_cuda:
        torch.cuda.empty_cache()