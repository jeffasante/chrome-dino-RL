import torch

from dqn_model import DQN, MLP
from utils.gym_dino_run import DinoRunEnv
from utils.extract import get_screen

def infer(env, mlp=False):
    
    action_size = env.action_space.n
    
    if mlp:
        policy_net = MLP(num_actions=action_size)
        
    else:
        policy_net = DQN()
        
        
    
    policy_net.load_state_dict(torch.load('saved_weights/Episode 2078-checkpoint.pth'))

    for i in range(3):
        env.reset()
        state = get_screen(env)

        for j in range(200):
            action =  policy_net(state).max(1)[1].view(1, 1)

            env.render()
            _, reward, done, _ = env.step(action.item())

            state = get_screen(env)
            if done:
                break

    env.close()
    


if __name__ == '__main__':
    # Get chrome dino run game
    env = DinoRunEnv()

    # initialize browser
    init = env.reset()

    state, reward, done, info = env.step(0)
    
    # Running Training
    infer(env=env)
    