import torchvision.transforms as T
from PIL import Image
import numpy


'''Input Extraction'''

def get_dino_location(screen_height, screen_width):
    world = screen_height * screen_width
    scale = screen_width / world
    
    return int(screen_height * scale + screen_width / 2.0)



resize = T.Compose([T.ToPILImage(),
                    T.Resize((80,80,), interpolation=Image.CUBIC),
                    T.ToTensor()])


def re(): return 'shit'

def get_screen(env):
    
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
     
    _, screen_height, screen_width  = screen.shape
    
    # strip of the top and bottom
    screen = screen[:, int(screen_height*0.2):int(screen_height*0.5), :]
    
    dino_loc = get_dino_location(screen_height, screen_width)
    
    screen = screen[:, :dino_loc, :]
    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen)
    
    
    return resize(screen).unsqueeze(0) 