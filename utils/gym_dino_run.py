# Custom OpenAI Gym environment
import gym
from gym import spaces, error

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from pyvirtualdisplay import Display 

import numpy as np
import os
import time
from io import BytesIO
from PIL import Image
import base64
import cv2



class GameInterface:
    
    def __init__(self, game_url='chrome://dino',
                 chromedrive='./utils/chrome_driver/chromedriver.exe'):
        
        '''Web Interface.'''
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--mute-audio')
        
        self.driver = webdriver.Chrome(chromedrive,
                                       chrome_options=chrome_options)
        self.driver.set_window_position(x=-10, y=0)
        
        
        try:
            self.driver.get(game_url)
        except:
            print('Running offline..')
            
            
    
    def press_up(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        
    def press_down(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
        
    def press_space(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.SPACE)

    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    
    def pause(self):
        return self.driver.execute_script("return Runner.instance_.stop()")
    
    def resume(self):
        return self.driver.execute_script("return Runner.instance_.play()")
    
    def end(self):
        self.driver.close()
        
        
        
        
class DinoRunEnv(gym.Env, GameInterface):
    
    
    def __init__(self, screen_width=120, screen_height=120, headless=False):
        
        gym.Env.__init__(self)
        GameInterface.__init__(self, game_url='chrome://dino')
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.driver.execute_script("Runner.config.ACCELERATION=0")
        
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self.driver.execute_script(init_script)
                
        # action_space and observation_space 
        self.action_space = spaces.Discrete(3) # do nothing, up, down
        
        self.observation_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (self.screen_width, self.screen_height, 3),
            dtype=np.uint8
        )
        
        
        self.viewer = None

        self.actions_lookup = {0:lambda: None,
                               1:self.press_up, 
                               2:self.press_down
                              }

        # All graphical operations are performed in virtual memory without showing any screen output.
        if headless:
            display = Display(visible=0, size=(1024, 768)) 
            display.start()
            
        
    def step(self, action):
        '''returns observaton, reward, done, extra_info'''

        
        assert action in self.action_space
        
        self.actions_lookup[action]()
        
        obs = self._get_screen()
        
        done, reward = (True, -1) if self._get_crashed() else (False, 0.1)
        
        time.sleep(.015)
    
        
        return obs, reward, done, {'score': self._get_score()}
        
    
    def render(self, mode='human'):
        ''' Return image array '''
        image = cv2.cvtColor(self._get_screen(), cv2.COLOR_BGR2RGB)
        
        if mode == 'rgb_array':
            return image
        
        elif mode =='human':
            
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            
            self.viewer.imshow(image)
            
            
            return self.viewer.isopen
            
        
    def reset(self):
        
        self.driver.execute_script("Runner.instance_.restart()")

        self.step(1)
        time.sleep(2)

        return self._get_screen()
    
                              
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    
         
    def _get_screen(self):
        
        image = self.driver.get_screenshot_as_base64()
        return np.array(Image.open(BytesIO(base64.b64decode(image))))
            

    
    
    def _get_score(self):
        return int(''.join \
            (self.driver.execute_script("return Runner.instance_.distanceMeter.digits")))
        
    
    
        
    
    def _get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    
