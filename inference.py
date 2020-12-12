import torch


policy_net.load_state_dict(torch.load('Episode 2078-checkpoint.pth'))

for i in range(3):
    env.reset()
    state = get_screen()
    
    for j in range(200):
        action =  policy_net(state).max(1)[1].view(1, 1)
        
        env.render()
        _, reward, done, _ = env.step(action.item())
        
        state = get_screen()
        if done:
            break
            
env.close()