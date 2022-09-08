import torch
import torch.nn as nn


def gradient_step(buffer, Q, target_Q, batch_size, device, gamma, optimizer):
    if(len(buffer) < 1000):
        return
    transitions = buffer.sample(batch_size)
    batch = buffer.transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    Q_values = Q(state_batch).gather(1, action_batch)

    next_Q_values = torch.zeros(batch_size, device=device)
    next_Q_values[non_final_mask] = target_Q(non_final_next_states).max(1)[0].detach()

    targets = (next_Q_values * gamma) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(Q_values, targets.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()