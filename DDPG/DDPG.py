import torch
import torch.nn as nn


def gradient_step(buffer, critic, target_critic, actor, target_actor, batch_size, device, gamma, critic_opt, actor_opt):
    if(len(buffer) < 10000):
        return

    target_critic.eval()
    target_actor.eval()

    transitions = buffer.sample(batch_size)
    batch = buffer.transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    ##########################################################

    action_batch = action_batch.view((batch_size,1))

    Q_values = critic(torch.cat((state_batch,action_batch),1))

    next_Q_values = torch.zeros(batch_size, device=device)

    next_a = target_actor(non_final_next_states).detach()

    next_Q_values[non_final_mask] = target_critic(torch.cat((non_final_next_states,next_a),1)).view(-1).detach()

    targets = (next_Q_values * gamma) + reward_batch


    criterion = nn.MSELoss()
    loss = criterion(Q_values, targets.unsqueeze(1))

    critic_opt.zero_grad()
    loss.backward()
    critic_opt.step()

    policy_loss = -torch.mean(critic(torch.cat((state_batch, actor(state_batch)), 1)))

    actor_opt.zero_grad()
    policy_loss.backward()
    actor_opt.step()