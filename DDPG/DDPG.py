import torch
import torch.nn as nn


def critic_step(buffer, critic, target_critic, target_actor, batch_size, device, gamma, optimizer):
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

    action_batch = action_batch.reshape ((128,1))

    Q_values = critic(torch.cat((state_batch,action_batch),1))

    next_Q_values = torch.zeros(batch_size, device=device)

    next_a = target_actor(non_final_next_states).detach()

    next_Q_values[non_final_mask] = target_critic(torch.cat((non_final_next_states,next_a),1)).view(-1).detach()

    targets = (next_Q_values * gamma) + reward_batch


    criterion = nn.MSELoss()
    loss = criterion(Q_values, targets.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def actor_step(buffer, critic, actor, batch_size, device, gamma, optimizer):
    if(len(buffer) < 10000):
        return

    transitions = buffer.sample(batch_size)
    batch = buffer.transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)

    policy_loss = -torch.mean(critic(torch.cat((state_batch,actor(state_batch)),1)))

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()