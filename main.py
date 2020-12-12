from utils.gym_dino_run import DinoRunEnv
from dqn_learn import train_dqn
from utils.replay_buffer import ReplayMemory
from dqn_model import DQN, MLP
from utils.extract import get_screen

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim


GAMMA = 0.99
BATCH_SIZE = 32
EPS_START = 0.01
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_episodes = 2 #0000




def main(env, mlp=False):
    
    action_size = env.action_space.n
    
    if mlp:
        policy_net = MLP(num_actions=action_size)
        target_net = MLP(num_actions=action_size)

    else:
        policy_net = DQN()
        target_net = DQN()
        
        

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025) # optim.Adam(policy_net.parameters(), lr=5e-4) 
    memory = ReplayMemory(int(1e5))

    
#     print('done')
    scores = train_dqn(env=env, policy_net=policy_net, target_net=target_net, memory=memory, GAMMA=GAMMA, BATCH_SIZE=BATCH_SIZE, EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, TARGET_UPDATE=TARGET_UPDATE,
                             n_episodes=n_episodes, get_screen=get_screen, optimizer=optimizer)
    
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()        
    print('Complete')
    
    
    
    
    

if __name__ == '__main__':
    
    # Get chrome dino run game
    env = DinoRunEnv()

    # initialize browser
    init = env.reset()

    state, reward, done, info = env.step(0)
    
    # Running Training
    main(env=env)
    
    
    