"""
This module contains the SailingEnv for discrete path planning with dynamic environment
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

from numpy import ndarray
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

from gym import Env, spaces

import numpy as np
from gym import Env, spaces
import pygame

class SailingEnv(Env):
    """   
    Sailing in a discrete world where agent seeks to reach goal with changing wind patterns. 
    
    This environment is based on that of `JonAsbury's Sailing-v0 <https://gist.github.com/JonAsbury/1a8102e070b1ad9888857e7cbcb48f93>`_
    
    For more information see `gym.Env docs <https://www.gymlibrary.dev/api/core/>`_
        
    **States** (dict)
    
        - "pose": [x, y, heading]
        - "wind": $m$ x $n$ np int array (values 0-7)
        
        where $m$ is the size of the x-dimension and $n$ the size in y.
        
    **Observations**
    
        Agent position is fully observable

    **Actions**
    
        - -1: turn left 45\u00b0
        -  0: move straight
        -  1: turn right 45\u00b0
    
    **Transition Probabilities**

        - agent moves in desired direction determininstically
        - $p$ probability of wind changing at *each* cell
        
    **Reward**
    
        $R = $
        
        - $R_{min}, \qquad \qquad \qquad \qquad \quad$ for hitting boundary
        - $R_{max}, \qquad \qquad \qquad \qquad \quad d = 0$,
        - $-0.01 - ||h - w||_2 - ||m - g||_2 + $

            - $-0.1, \qquad \qquad \qquad \qquad \quad$ when $\leq 5$ cells from boundary
            - $(R_{max}-100)(1 - \dfrac{d}{r_{goal}}^2), \; d \leq r_{goal}$
        
    
        where 
        
        - $m$ is the movement direction normalized to $\sqrt{2}$
        - $w$ is the wind direction normalized to $\sqrt{2}$
        - $g$ is the goal direction normalized to $\sqrt{2}$
        - $d$ is the distance to the goal
        - $r_{goal}$ is the reward radius of the goal, and
        - $R_i$ are the reward extrema.

    **Input**
    
    :param seed: (int) RNG seed, *default*: None
    
    Remaining parameters are passed as arguments through the ``params`` dict.
    The corresponding keys are as follows:
    
    :param dimensions: ([x,y]) size of map, *default* [40,40]
    :param goal: ([x,y]) position of goal, *default* [10,10]
    :param state: (State) Initial state (wind not required), *default*: {"pose": [20,20]}, wind undefined
    :param p: (float) probability of wind changing at each cell, *default*: 0.1
    :param r_radius: (float) Reward radius, *default*: 5.0
    :param r_range: (tuple) min and max params of reward, *default*: (-400, 1100)
    :param render: (str) render mode (see metadata for options), *default*: "none"
    :param cell_size: (int) size of cells for visualization, *default*: 5
    :param prefix: (string) where to save images, *default*: "<cwd>/plot"
    :param save_frames: (bool) save images for gif, *default*: False
    :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
    """
    metadata = {"render_modes": ["plot", "print", "none"], "render_fps": 5}

    def __init__(self, *, seed : int = None, params : dict = None):
        super(SailingEnv, self).__init__()
        
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
        self._log.debug("Init Sailing")
        
        self._params = {}
        self._state = {}
        self.reset(seed=seed, options=params)
        
        self._id_action = {
            0: np.array([ 0, -1]),
            1: np.array([-1, -1]),
            2: np.array([-1,  0]),
            3: np.array([-1,  1]),
            4: np.array([ 0,  1]),
            5: np.array([ 1,  1]),
            6: np.array([ 1, 0]),
            7: np.array([ 1, -1]),
        }
        
        self.action_space = spaces.discrete.Discrete(3, start=-1)

        upper = self._params["dimensions"].copy()
        upper.append(8)
        
        self.observation_space = spaces.Dict(
            {
                "pose": spaces.box.Box(low=np.zeros(3), high=np.array(upper)-1, dtype=int),
                "wind": spaces.box.Box(low=np.zeros(self._params["dimensions"]), high=7*np.ones(self._params["dimensions"]), dtype=int)
            }
        )
        
        self._triangle = [np.array([0.3,0]),np.array([-0.3,-0.15]),np.array([-0.3,0.15])]
        for i, el in enumerate(self._triangle):
            self._triangle[i] = el*self._params["cell_size"]
    
    def reset(self, *, seed: int = None, options: dict = {}):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*: {}
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        super().reset(seed=seed)
        self._log.debug("Reset Sailing")
        
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
                self._params["state"]["pose"] = [np.round(self._params["dimensions"][0]/2), np.round(self._params["dimensions"][1]/2), self.np_random.integers(0,8)]
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
        
        # Potential TODO add option to retain wind at reinit
        if "wind" not in options:
            self._sample_wind(True)
            self._params["state"]["wind"] = deepcopy(self._state["wind"])
        else:
            self._state["wind"] = self._params["state"]["wind"]
        
        self._state = deepcopy(self._params["state"])
        self._log.info(str(self._state))  
        
        return self._get_obs(), self._get_info()
    
    def _sample_wind(self, is_new : bool = False):
        """
        Samples the wind in the environment
        
        - if nonexistent or is_new is true environment will be sample from scratch
        - else for each state, with probability $p$ (from ``params``), uniformly rotate wind by $\pm$ 45\u00b0 
        
        :param is_new: (bool) is new simulation, *default*: False
        """
        if "wind" not in self._state or is_new:
            self._log.debug("Resample wind from scratch")
            self._state["wind"] = np.zeros(self._params["dimensions"])
            for i in range(self._params["dimensions"][0]):
                for j in range(self._params["dimensions"][1]):
                    self._state["wind"][i][j] = self.np_random.choice(range(8))
        else:
            self._log.debug("Resample wind update")
            for i in range(self._params["dimensions"][0]):
                for j in range(self._params["dimensions"][1]):
                    p = self.np_random.uniform()
                    if p < self._params["p"]:
                        dir =  self.np_random.choice([-1,1])
                        dir = self._state["wind"][i][j] + dir
                        if dir < 0:
                            dir = 7
                        elif dir > 7:
                            dir = 0
                        self._state["wind"][i][j] = dir
    
    def step(self, a : int):
        """
        Increments enviroment by one timestep 
        
        :param a: (int) action, *default*: None
        :return: (tuple) State, reward, is_done, is_truncated, info 
        """
        self._log.debug("Step action " + str(a))
        
        s = deepcopy(self._state)
        
        self._state["pose"][2] = self._update_heading(self._state["pose"][2], a)
        
        p1 = deepcopy(self._state)
        p1["pose"][0:2] += self._id_action[self._state["pose"][2]]
        
        if self.observation_space.contains(p1):
            print("yee")
            self._state["pose"] = p1.copy()
        done = False
        if np.all(self._state["pose"] == self._params["goal"]):
            done = True        
        
        self._sample_wind()
        
        r = self.reward(s, a, self._state)       
        self._log.info("Is terminal: " + str(done) + ", reward: " + str(r))    
        return self._get_obs(), r, done, False, self._get_info()   
    
    def _update_heading(self, heading : int, a : int):
        """
        Updates heading of a given state, keeping it within bounds

        :param pose: (ndarray) pose to update
        :param a: (int) action to update
        :return: (int) heading
        """
        heading += a
        if heading < 0:
            heading = 7
        if heading > 7:
            heading = 0
        return heading

    def get_actions(self, s : dict):
        """
        Gets list of actions for a given pose

        :param s: (State) state from which to get actions
        :return: ((list) actions, (list(ndarray)) subsequent poses without wind)
        """
        self._log.debug("Get Actions at state : " + str(s))
        neighbors = []
        actions = []
        state = deepcopy(s)
        pose = state["pose"].copy()
        
        for i, el in enumerate([-1,0,1]):
            state["pose"][2]   = self._update_heading(state["pose"][2], el)
            state["pose"][0:2] = pose[0:2] + self._id_action[state["pose"][2]]
            
            if self.observation_space.contains(state):
                neighbors.append({"pose": state["pose"]})
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
        information = {"distance": np.linalg.norm(self._state["pose"][0:2] - self._params["goal"])}
        self._log.debug("Get Info: " + str(information))
        return information

    def reward(self, s : dict, a : int = None, sp : dict = None):
        """
        Gets rewards for $(s,a,s')$ transition
        
        :param s: (State) Initial state
        :param a: (int) Action (unused in this environment), *default*: None
        :param sp: (State) resultant state, *default*: None
        :return: (float) reward 
        """
        # reef
        if int(sp["pose"][0]) == 0 or int(sp["pose"][0]) >= (self._params["dimensions"][0]-1) or int(sp["pose"][1]) == 0 or int(sp["pose"][1]) >= (self._params["dimensions"][1]-1):
            return self.reward_range[0]
        
        goal_direction = self._params["goal"] - sp["pose"][0:2]
        distance = np.linalg.norm(goal_direction) 
        # goal
        if distance == 0:
            return self.reward_range[1]

        # time penalty
        r = -0.01

        if s != [] and a != []: # This is done for rendering
            # wind penalty
            wind_direction = s["wind"][int(s["pose"][0])][int(s["pose"][1])]
            r -= np.linalg.norm(self._id_action[wind_direction] - self._id_action[sp["pose"][2]])
            
        # goal direction penalty
        goal_direction = goal_direction * np.sqrt(2) / distance # normalizes direction to sqrt(2)
        r -= np.linalg.norm(goal_direction - self._id_action[sp["pose"][2]])
        
        # shoals  
        if sp["pose"][0] < 10 or sp["pose"][0] > self._params["dimensions"][0]-10 or sp["pose"][1] < 10 or sp["pose"][1] > self._params["dimensions"][1]-10:
            r -= 0.1
        
        # goal radius 
        if distance <= self._params["r_radius"] and distance > 0:
            r += (self.reward_range[1]-100)*(1 - (distance/self._params["r_radius"])**2 )
        
        return r
    
    def render(self):
        """    
        Renders environment
        
        Has two render modes: 
        
        - *plot* uses PyGame visualization
        - *print* logs pose at Warning level

        Visualization
        
        - blue triangle: agent
        - green diamond: goal 
        - red diamond: goal + agent
        - orange triangle: wind direction
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
                    r = self.reward([],[],{"pose": [i,j,self._state["wind"][i][j]]})
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
                print("----------")
                move_direction = self._id_action[self._state["pose"][2]]
                print(move_direction)
                move_direction = np.arctan2(move_direction[1],move_direction[0])
                print(move_direction)
                print("mem tri ", self._triangle)
                triangle = self._rotate_polygon(self._triangle,move_direction)
                print(triangle)
                print("pose ", self._state["pose"]*self._params["cell_size"])
                for i, el in enumerate(triangle):
                    triangle[i] = 1.2*el + (self._state["pose"][0:2]+0.5)*self._params["cell_size"]
                print(triangle)
                pygame.draw.polygon(img, (0,0,255), triangle)
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
            p = self._state["pose"]
            self._log.warning("Pose " + str(self._state["pose"]) + " | wind " + str([self._state["wind"][p[0]],self._state["wind"][p[1]]]))
            
            255, 83, 73
    
    def _rotate_polygon(self, vertices : list, angle : float, center : ndarray = np.zeros(2)):
        """
        Rotates a polygon by a given angle

        :param vertices: (list(ndarray)) List of 2d coordinates
        :param angle: (float) angle in radians to rotate polygon
        :param center: (ndarray) coordinate about which to rotate polygon, *default*: [0,0] 
        """
        # Since there are only a few angles, could potentially preload all of them them pass in the matrix
        vertices = deepcopy(vertices)
        R = np.zeros([2,2])
        R[0,0] = np.cos(angle)
        R[0,1] = -np.sin(angle)
        R[1,0] = np.sin(angle)
        R[1,1] = np.cos(angle)
        for i in range(len(vertices)):
            vertices[i] -= center
            vertices[i]  = np.matmul(R,vertices[i])
            vertices[i] += center
        return vertices