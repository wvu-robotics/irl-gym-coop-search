"""
This module contains the GridworldEnv for discrete path planning
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

from copy import deepcopy
from typing import Optional
import logging

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt

from gym import Env, spaces
import pygame

from irl_gym.utils.utils import *

class GridWorldEnv(Env):
    """   
    Simple Gridworld where agent seeks to reach goal. 
    
    For more informatin see `gym.Env docs <https://www.gymlibrary.dev/api/core/>`_
        
    **States** (dict)
    
    - "pose": [x,y]
        
    **Observations**
    
    Agent position is fully observable

    **Actions**
    
    - 0: move south        [ 0, -1]
    - 1: move west         [-1,  0]
    - 2: move north        [ 0,  1]
    - 3: move east         [ 1,  0]
    
    **Transition Probabilities**

    - $p \qquad \qquad$ remain in place
    - $1-p \quad \quad \:$ transition to desired state
        
    **Reward**
    
    - $R_{min} \qquad \qquad d > r_{goal} $
    
    - $R_{max} - \dfrac{d}{r}^2 \quad \; d \leq r$
    
    where $d$ is the distance to the goal, $r_{goal}$ is the reward radius of the goal, and
    R_i$ are the reward extrema.
    
    **Input**
    :param seed: (int) RNG seed, *default*: None
    
    Remaining parameters are passed as arguments through the ``params`` dict.
    The corresponding keys are as follows:
    
    :param dimensions: ([x,y]) size of map, *default* [40,40]
    :param goal: ([x,y]) position of goal, *default* [10,10]
    :param state: (State) Initial state, *default*: {"pose": [20,20]}
    :param r_radius: (float) Reward radius, *default*: 5.0
    :param p: (float) probability of remaining in place, *default*: 0.1
    :param r_range: (tuple) min and max params of reward, *default*: (-0.01, 1)
    :param render: (str) render mode (see metadata for options), *default*: "none"
    :param cell_size: (int) size of cells for visualization, *default*: 5
    :param prefix: (string) where to save images, *default*: "<current>/irl_gym/plot"
    :param save_frames: (bool) save images for gif, *default*: False
    :param log_level: (int) Level of logging to use. For more info see `logging levels<https://docs.python.org/3/library/logging.html#levels>`, *default*: warning
    """
    metadata = {"render_modes": ["plot", "print", "none"], "render_fps": 5}

    def __init__(self, *, seed = None, params = {}):
        super(GridWorldEnv, self).__init__()
        
        self._log = logging.getLogger( __name__)
        if "log_level" not in params:
            self._log.setLevel(logging.WARNING)
        else:
            self._log.setLevel(params["log_level"])
        self._log.debug("Init GridWorld")
        
        self.reset(seed, options=params)
        
        self._id_action = {
            0: np.array([0, -1]),
            1: np.array([-1, 0]),
            2: np.array([0, 1]),
            3: np.array([1, 0]),
            4: np.array([0, 0]),
        }
        
        self.action_space = spaces.discrete.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "pose": spaces.box.Box(low =np.zeros(2), high=np.array(self._params["dimensions"])-1, dtype=int)
            }
        )
    
    def reset(self, *, seed: int = None, options: Optional[dict] = None):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*:, None
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        super().reset(seed=seed)
        self._log.debug("Reset GridWorld")
        
        if options != None:
            for el in options:
                self._params[el] = options[el]
        
            if "dimensions" not in self._params:
                self._params["dimensions"] = [40, 40]
            if "goal" not in self._params:
                self._params["goal"] = [np.round(self._params["dimensions"][0]/4), np.round(self._params["dimensions"][1]/4)]
            if "state" not in self._params:
                self._params["state"]["pose"] = [np.round(self._params["dimensions"][0]/2), np.round(self._params["dimensions"][1]/2)]
                print("p",type(self._params["state"]["pose"]))
            if "r_radius" not in self._params:
                self._params["r_radius"] = 5
            if "r_range" not in self._params:
                self.reward_range((-0.01, 1))
            if "p" not in self._params:
                self._params["p"] = 0.1
            if "render" not in self._params:
                self._params["render"] = "none"
            if "print" not in self._params:
                self._params["print"] = False
            if self._params["render"] == "plot":
                self.fig_ = plt.figure()
                self.ax_ = self.fig_.add_subplot(1,1,1)
                self.map_ = np.zeros(self._params["dimensions"])
                for i in range(self._params["dimensions"][0]):
                    for j in range(self._params["dimensions"][1]):
                        self.map_[i][j] = self.reward({"pose":[i,j]})
                self.window = None
                self.clock = None
                if "cell_size" not in self._params:
                    self._params["cell_size"] = 5
            if "save_frames" not in self._params:
                self._params["save_frames"] = False
            if "prefix" not in self._params:
                self._params["prefix"] = current   
            if self._params["save_frames"]:
                self._img_count = 0   
        
        if type(self._params["state"]["pose"]) != np.ndarray:
            self._params["state"]["pose"] = np.array(self._params["state"]["pose"])
        self._state = self._params["state"]      

        self._log.info(str(self._state))        
        return self._get_obs(), self._get_info()
    
    def step(self, _action):
        """
        Increments enviroment by one timestep 
        
        :param _action: (int) random number generator seed, *default*: None
        :return: (tuple) State, reward, is_done, is_truncate, info 
        """
        self._log.debug("Step action " + str(_action))
        
        done = False
        
        if self.np_random.multinomaial(1,[self._params["p"], 1-self._params["p"]]):
            # multinomial produces a 1, then we got 1-p outomce
            # so carry out action, otherwise nothing happens
            p1 = self._state["pose"].copy()
            p1 += self._id_action(_action)
            
            if self.observation_space.contains(p1):
                self._state["pose"] = p1
        
            if np.all(self._state["pose"] == self._params["goal"]):
                done = True
        
        r = self.reward(self._state)
        self._log.info("Is terminal: " + str(done) + ", reward: ", r)    
        return self._get_obs(), r, done, False, self._get_info()
    
                          
    def get_actions(self, state):
        """
        Gets list of actions for a given pose

        :param state: (State) state from which to get actions
        :return: ((list) actions, (list(ndarray)) subsequent states)
        """
        neighbors = []
        actions = []
        position = state["pose"].copy()

        for i, el in enumerate(self._id_action):
            temp = position.copy()
            temp += self._id_action[el]
            
            if self.observation_space.contains(temp):
                neighbors.append({"pose": temp})
                actions.append(i)
                
        return actions, neighbors
    
    def _get_obs(self):
        """
        Gets observation
        
        :return: (State)
        """
        self._log.debug("Get Obs: " + str(self._state))
        return deepcopy(self._state)
    
    def _get_info(self):
        """
        Gets info on system
        
        :return: (dict)
        """
        information = {"distance": np.linalg.norm(self._state["pose"] - self._params["goal"], ord=1)}
        self._log.debug("Get Obs: " + str(information))
        return information
    
    def reward(self, _s, _a = None, _sp = None):
        """
        Gets rewards for $(s,a,s')$ transition
        
        :param _s: (State) Initial state
        :param _a: (int) Action (unused in this class), *default*: None
        :param _sp: (State) Action (unused in this class), *default*: None
        :return: (float) reward 
        """
        self._log.debug("Get reward")
        d = np.linalg.norm(self._state["pose"] - self._params["goal"], ord=1)
        if d >= self._params["r_radius"]:
            return self.reward_range[0]
        else:
            return self.reward_range[1] - (d/self._params["r_radius"])**2

    def render(self):
        """    
        Renders environment
        
        Has two render modes: 
        
        - *plot* uses PyGame visualization
        - *print* logs state to console

        Visualization
        
        - blue circle: agent
        - green diamond: goal 
        - red diamond: goal + agent
        """
        if self._params["render_mode"] == "plot":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self._params["dimensions"][0]*self._params["cell_size"], self._params["dimensions"][1]*self._params["cell_size"]))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            img = pygame.Surface((self._params["dimensions"][0]*self._params["cell_size"], self._params["dimensions"][1]*self._params["cell_size"]))
            img.fill((255,255,255))
            
            # Agent
            if np.all(self._state["pose"] == self._params["goal"]):
                pygame.draw.polygon(img, (255,0,0), [(self._state["pose"]+np.array([0.75,0])), (self._state["pose"]+np.array([0,0.75])), 
                                                     (self._state["pose"]+np.array([-0.25,0])), (self._state["pose"]+np.array([0,-0.25]))])
            else:
                pygame.draw.circle(img, (0,0,255), (self._state["pose"]+0.5)*self._params["cell_size"], self._params["cell_size"]/2)
                pygame.draw.polygon(img, (0,255,0), [(self._state["pose"]+np.array([0.75,0])), (self._state["pose"]+np.array([0,0.75])), 
                                                     (self._state["pose"]+np.array([-0.25,0])), (self._state["pose"]+np.array([0,-0.25]))])
            
            for x in range(self._params["dimensions"][1]):
                pygame.draw.line(img, 0, (0, self._params["cell_size"] * x), (self.window_size, self._params["cell_size"] * x), width=3)
            for y in range(self._params["dimensions"][1]):
                pygame.draw.line(img, 0, (self._params["cell_size"] * y, 0), (self._params["cell_size"] * y, self.window_size), width=3)
                
            self.window.blit(img, img.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
            if self._params["save_frames"]:
                pygame.image.save(img, self.prefix_ + "img" + str(self.count_im_) + ".png")
                self.count_im_+=1
                
        elif self._params["render_mode"] == "print":
            self._log.warning(str(self._state))
