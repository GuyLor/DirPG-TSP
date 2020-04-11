def log_values_dirpg(to_log, grad_norms, epoch, batch_id, step, writer, opts):
    avg_cost_opt = to_log['opt_cost'].mean().item()
    avg_cost_direct = to_log['direct_cost'].mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}'.format(epoch, batch_id))
    print('avg_cost opt: {}, avg_cost direct: {}'.format(avg_cost_opt, avg_cost_direct))
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        print('============================')
        # writer.add_scalar('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])
        # writer.add_scalar('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])

        writer.add_scalars('cost', {'opt': avg_cost_opt, 'direct': avg_cost_direct}, to_log['interactions'])
        writer.add_scalar('grad_norm', grad_norms[0], step)
        writer.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('cost', avg_cost, step*opts.graph_size)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step*opts.graph_size)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)