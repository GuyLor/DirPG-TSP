import numpy as np
import torch
def log_values_dirpg(to_log, grad_norms, epoch, batch_id, step, writer, opts):
    avg_cost_opt = to_log['opt_cost'] #.mean().item()
    avg_cost_direct = to_log['direct_cost'] #.mean().item()
    grad_norms, grad_norms_clipped = grad_norms
    interactions = to_log['interactions'].item()
    # Log values to screen
    print('epoch: {}, train_batch_id: {}, interactions: {}'.format(epoch, batch_id, interactions))
    print('avg_cost opt: {:.4f}, avg_cost direct: {:.4f}'.format(avg_cost_opt, avg_cost_direct))
    print('avg_objective opt: {:.4f}, avg_objective direct: {:.4f}'.format(to_log['opt_objective'], to_log['direct_objective']))
    print('candidates: {:.3f},'
          '  prune_count: {:.3f},'
          '  empty heaps: {:.3f},'
          ' interactions: {:.3f}'.format(to_log['candidates'],
                                         to_log['prune_count'],
                                         to_log['empty_heaps'],
                                         to_log['interactions_count']))
    print('grad_norm: {:.4f}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:

        # writer.add_scalar('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])
        # writer.add_scalar('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])

        writer.add_scalars('cost/interactions', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, interactions)
        writer.add_scalars('cost/steps', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, step)

        writer.add_scalar('empty heaps ratio', to_log['empty_heaps']/opts.batch_size, interactions)
        writer.add_scalar('number of candidates', to_log['candidates'], interactions)

        writer.add_scalar('interactions until improvement', to_log['interactions_count'], step)

        writer.add_scalar('grad_norm', grad_norms[0], interactions)
        writer.add_scalar('grad_norm_clipped', grad_norms_clipped[0], interactions)

        #writer.add_scalars('search', {'dfs': to_log['dfs'], 'bfs': to_log['bfs'], 'jumps': to_log['jumps']}, step)
        #writer.add_scalar('candidates', to_log['candidates'], step)

def log_values_supervised(to_log, grad_norms, epoch, batch_id, step, writer, opts):
    avg_cost_opt = to_log['opt_cost'] #.mean().item()
    avg_cost_direct = to_log['direct_cost'] #.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    interactions = to_log['interactions'].cpu().item()
    print('epoch: {}, train_batch_id: {}, interactions: {}'.format(epoch, batch_id,interactions ))
    print('avg_cost opt: {:.4f}, avg_cost direct: {:.4f}'.format(avg_cost_opt, avg_cost_direct))
    print('avg_objective opt: {:.4f}, avg_objective direct: {:.4f}'.format(to_log['opt_objective'], to_log['direct_objective']))
    print('grad_norm: {:.4f}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:

        # writer.add_scalar('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])
        # writer.add_scalar('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])

        writer.add_scalars('cost/interactions', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, interactions)
        writer.add_scalars('cost/steps', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, step)
        writer.add_scalar('objective', to_log['opt_objective'], interactions)
        writer.add_scalar('nll', to_log['direct_objective'], interactions)

        #writer.add_scalars('search', {'dfs': to_log['dfs'], 'bfs': to_log['bfs'], 'jumps': to_log['jumps']}, step)
        #writer.add_scalar('candidates', to_log['candidates'], step)

def log_values(cost, grad_norms, epoch, interactions, batch_id, step,
               log_likelihood, opt_nll, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    #interactions = 2 * step * opts.graph_size * opts.batch_size if opts.baseline is not None else step * opts.graph_size * opts.batch_size
    interactions = interactions.to(torch.int64).cpu().item()
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))
    print('interactions: {}, nll: {}, log_p: {}'.format(interactions, -opt_nll.mean().item(), -log_likelihood.mean().item()))
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard


    if not opts.no_tensorboard:
        tb_logger.log_value('cost/interactions', avg_cost, interactions )
        tb_logger.log_value('cost/steps', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('objective', -log_likelihood.mean().item(),interactions)
        tb_logger.log_value('nll', -opt_nll.mean().item(), interactions)

        tb_logger.log_value('grad_norm', grad_norms[0], interactions)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], interactions)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)


