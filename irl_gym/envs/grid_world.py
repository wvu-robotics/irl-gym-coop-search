"""
This module contains the GridworldEnv for discrete path planning
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

import numpy as np
from gym import Env, spaces
import pygame

class GridWorldEnv(Env):
    """   
    Simple Gridworld where agent seeks to reach goal. 
    
    For more information see `gym.Env docs <https://www.gymlibrary.dev/api/core/>`_
        
    **States** (dict)
    
        - "pose": [x, y]
        
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
    
        - $R_{min}, \qquad \qquad \quad d > r_{goal} $
        - $R_{max} - \dfrac{d}{r_{goal}}^2, \quad d \leq r_{goal}$
    
        where $d$ is the distance to the goal, $r_{goal}$ is the reward radius of the goal, and
        R_i$ are the reward extrema.
    
    **Input**
    
    :param seed: (int) RNG seed, *default*: None
    
    Remaining parameters are passed as arguments through the ``params`` dict.
    The corresponding keys are as follows:
    
    :param dimensions: ([x,y]) size of map, *default* [40,40]
    :param goal: ([x,y]) position of goal, *default* [10,10]
    :param state: (State) Initial state, *default*: {"pose": [20,20]}
    :param p: (float) probability of remaining in place, *default*: 0.1
    :param r_radius: (float) Reward radius, *default*: 5.0
    :param r_range: (tuple) min and max params of reward, *default*: (-0.01, 1)
    :param render: (str) render mode (see metadata for options), *default*: "none"
    :param cell_size: (int) size of cells for visualization, *default*: 5
    :param prefix: (string) where to save images, *default*: "<cwd>/plot"
    :param save_frames: (bool) save images for gif, *default*: False
    :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
    """
    metadata = {"render_modes": ["plot", "print", "none"], "render_fps": 5}

    def __init__(self, *, seed : int = None, params : dict = None):
        super(GridWorldEnv, self).__init__()
        
        self._log = logging.getLogger(__name__)
        log_sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_sh.setFormatter(formatter)
        self._log.addHandler(log_sh)

        if "log_level" not in params:
            self._log.setLevel(logging.WARNING)
        else:
            log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
            self._log.setLevel(log_levels[params["log_level"]])
        self._log.debug("Init GridWorld")
        
        self._params = {}
        self.reset(seed=seed, options=params)
        
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
                "pose": spaces.box.Box(low=np.zeros(2), high=np.array(self._params["dimensions"])-1, dtype=int)
            }
        )
    
    def reset(self, *, seed: int = None, options: dict = {}):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*:, {}
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        super().reset(seed=seed)
        self._log.debug("Reset GridWorld")
        
        if options != {}:
            for el in options:
                self._params[el] = deepcopy(options[el])
        
            if "dimensions" not in self._params:
                self._params["dimensions"] = [40, 40]
            if "goal" not in self._params:
                self._params["goal"] = [np.round(self._params["dimensions"][0]/4), np.round(self._params["dimensions"][1]/4)]
            if type(self._params["goal"]) != np.ndarray:
                self._params["goal"] = np.array(self._params["goal"], dtype = int)
            if "state" not in self._params:
                self._params["state"] = {"pose": None}
                self._params["state"]["pose"] = [np.round(self._params["dimensions"][0]/2), np.round(self._params["dimensions"][1]/2)]
            if type(self._params["state"]["pose"]) != np.ndarray:
                self._params["state"]["pose"] = np.array(self._params["state"]["pose"], dtype = int)
            if "r_radius" not in self._params:
                self._params["r_radius"] = 5
            if "r_range" not in self._params:
                self.reward_range = (-0.01, 1)
            else:
                self.reward_range = self._params["r_range"]
            if "p" not in self._params:
                self._params["p"] = 0.1
            if "render" not in self._params:
                self._params["render"] = "none"
            if "print" not in self._params:
                self._params["print"] = False
            if self._params["render"] == "plot":
                self.window = None
                self.clock = None
                if "cell_size" not in self._params:
                    self._params["cell_size"] = 5
            if "save_frames" not in self._params:
                self._params["save_frames"] = False
            if "prefix" not in self._params:
                self._params["prefix"] = os.getcwd() + "/plot/"
            if self._params["save_frames"]:
                self._img_count = 0   
        
        self._state = deepcopy(self._params["state"])
        self._log.info(str(self._state))
                
        return self._get_obs(), self._get_info()
    
    def step(self, a : int):
        """
        Increments enviroment by one timestep 
        
        :param a: (int) action, *default*: None
        :return: (tuple) State, reward, is_done, is_truncated, info 
        """
        self._log.debug("Step action " + str(a))
        
        done = False
        s = deepcopy(self._state)
        if self.np_random.multinomial(1,[self._params["p"], 1-self._params["p"]])[1]:
            # multinomial produces a 1, then we got 1-p outomce
            # so carry out action, otherwise nothing happens
            p1 = self._state["pose"].copy()
            p1 += self._id_action[a]
            
            if self.observation_space.contains({"pose": p1}):
                self._state["pose"] = p1
            if np.all(self._state["pose"] == self._params["goal"]):
                done = True
        
        r = self.reward(s, a, self._state)
        self._log.info("Is terminal: " + str(done) + ", reward: " + str(r))    
        return self._get_obs(), r, done, False, self._get_info()
    
                          
    def get_actions(self, s : dict):
        """
        Gets list of actions for a given pose

        :param s: (State) state from which to get actions
        :return: ((list) actions, (list(ndarray)) subsequent states)
        """
        self._log.debug("Get Actions at state : " + str(s))
        neighbors = []
        actions = []
        position = s["pose"].copy()

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
        information = {"distance": np.linalg.norm(self._state["pose"] - self._params["goal"])}
        self._log.debug("Get Info: " + str(information))
        return information
    
    def reward(self, s : dict, a : int = None, sp : dict = None):
        """
        Gets rewards for $(s,a,s')$ transition
        
        :param s: (State) Initial state (unused in this environment)
        :param a: (int) Action (unused in this environment), *default*: None
        :param sp: (State) resultant state, *default*: None
        :return: (float) reward 
        """
        self._log.debug("Get reward")
        d = np.linalg.norm(sp["pose"] - self._params["goal"])
        if d >= self._params["r_radius"]:
            return self.reward_range[0]
        else:
            return self.reward_range[1] - (d/self._params["r_radius"])**2

    def render(self):
        """    
        Renders environment
        
        Has two render modes: 
        
        - *plot* uses PyGame visualization
        - *print* logs state at Warning level

        Visualization
        
        - blue circle: agent
        - green diamond: goal 
        - red diamond: goal + agent
        - Grey cells: The darker the shade, the higher the reward
        """
        self._log.debug("Render " + self._params["render"])
        if self._params["render"] == "plot":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self._params["dimensions"][0]*self._params["cell_size"], self._params["dimensions"][1]*self._params["cell_size"]))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            img = pygame.Surface((self._params["dimensions"][0]*self._params["cell_size"], self._params["dimensions"][1]*self._params["cell_size"]))
            img.fill((255,255,255))
            for i in range(self._params["dimensions"][0]):
                for j in range(self._params["dimensions"][1]):
                    r = self.reward([],[],{"pose": [i,j]})
                    if r > 0:
                        pygame.draw.rect(img, ((1-r)*255, (1-r)*255, (1-r)*255), pygame.Rect(i*self._params["cell_size"], j*self._params["cell_size"], self._params["cell_size"], self._params["cell_size"]))
            
            goal = [(self._params["goal"]+np.array([ 1  , 0.5  ]))*self._params["cell_size"], 
                    (self._params["goal"]+np.array([ 0.5, 1  ]))*self._params["cell_size"], 
                    (self._params["goal"]+np.array([ 0,   0.5]))*self._params["cell_size"], 
                    (self._params["goal"]+np.array([ 0.5, 0  ]))*self._params["cell_size"]]
            # Agent
            if np.all(self._state["pose"] == self._params["goal"]):
                pygame.draw.polygon(img, (255,0,0), goal)
            else:
                pygame.draw.circle(img, (0,0,255), (self._state["pose"]+0.5)*self._params["cell_size"], self._params["cell_size"]/2)
                pygame.draw.polygon(img, (0,255,0), goal)
            
            for y in range(self._params["dimensions"][1]):
                pygame.draw.line(img, 0, (0, self._params["cell_size"] * y), (self._params["cell_size"]*self._params["dimensions"][0], self._params["cell_size"] * y), width=2)
            for x in range(self._params["dimensions"][0]):
                pygame.draw.line(img, 0, (self._params["cell_size"] * x, 0), (self._params["cell_size"] * x, self._params["cell_size"]*self._params["dimensions"][1]), width=2)
                
            self.window.blit(img, img.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
            if self._params["save_frames"]:
                pygame.image.save(img, self._params["prefix"] + "img" + str(self._img_count) + ".png")
                self._img_count += 1
                
        elif self._params["render"] == "print":
            self._log.warning(str(self._state))
