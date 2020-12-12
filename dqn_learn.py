from collections import deque
from itertools import count

import torch
import torch.nn.functional as F

import random
import math

import numpy as np



def learn(memory, BATCH_SIZE, policy_net, target_net, optimizer):

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    #This converts batch-array of Transitions to Transition of batch-arrays.
    batch = memory.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


    
    
steps_done = 0

def select_epilson_greedy_action(state, policy_net, BATCH_SIZE,EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE):
    global steps_done
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
       
        with torch.no_grad():
            
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]],
                            dtype=torch.long)
       
    
def train_dqn(env, policy_net, target_net, memory, GAMMA, 
                             BATCH_SIZE,EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, n_episodes, get_screen, optimizer):
   
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    
    print()
    for i_episode in range(n_episodes):
        # Initialize the environment and state
        score = 0

        env.reset()
        state = get_screen(env)

        for t in count():

            action = select_epilson_greedy_action(state, policy_net, BATCH_SIZE,EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE)
            _, reward, done, _ = env.step(action.item())

            reward = torch.tensor([reward],  dtype=torch.float)

            score += reward

            # Observe new state
            if not done:
                next_state= get_screen(env)
            else:
                next_state = None

            # store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state 

            # Perform one step of Optimization (on the target network)
            learn(memory, BATCH_SIZE, policy_net, target_net, optimizer)

            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        # Update the target netwrk, copying all weights and biases  in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Show stats
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window)>=20.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(policy_net.state_dict(), f'{n_episodes}-checkpoint.pth')
            break
            
    
    return scores
















    