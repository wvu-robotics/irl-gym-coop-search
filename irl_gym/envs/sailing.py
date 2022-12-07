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
    :param r_radius: (float) Reward radius, *default*: 5.0
    :param p: (float) probability of wind changing at each cell, *default*: 0.1
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
    
    def reset(self, *, seed: int = None, options: dict = None):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*:, None
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        super().reset(seed=seed)
        self._log.debug("Reset Sailing")
        
        if options != None:
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
            self._state["wind"] = np.zeros(self.dim_)
            for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    self._state["wind"][i][j] = self.rng_.choice(range(8))
        else:
            self._log.debug("Resample wind update")
            for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    p = self.rng_.uniform()
                    if p < self.p_:
                        dir =  self.rng_.choice([-1,1])
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
            self._state["pose"] = p1.copy()
        done = False
        if np.all(self._state["pose"] == self._params["goal"]):
            done = True        
        
        self._sample_wind()
        
        r = self.reward(s, a, self._state)       
        self._log.info("Is terminal: " + str(done) + ", reward: " + str(r))    
        return self.get_observation(), r, done, False, {}    
    
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
        if int(sp["pose"][0]) == 0 or int(sp["pose"][0]) >= (self.dim_[0]-1) or int(sp["pose"][1]) == 0 or int(sp["pose"][1]) >= (self.dim_[1]-1):
            return self.reward_range[0]
        
        d = self._params["goal"] - s[0:2]
        # goal
        if d == 0:
            return self.reward_range[1]

        # time penalty
        r = -0.01
        
        # wind penalty
        wind_direction = s["wind"][int(s["pose"][0])][int(s["pose"][1])]
        r -= np.linalg.norm(self._id_action[wind_direction] - self._id_action[sp["pose"][2]])
        
        # goal direction penalty
        goal_direction = d * np.sqrt(2) / np.linalg.norm(d) # normalizes direction to sqrt(2)
        r -= np.linalg.norm(goal_direction - self._id_action[sp["pose"][2]])
        
        # shoals  
        if sp["pose"][0] < 10 or sp["pose"][0] > self.dim_[0]-10 or sp["pose"][1] < 10 or sp["pose"][1] > self.dim_[1]-10:
            r -= 0.1
        
        # goal radius 
        if d <= self._params["r_radius"] and d > 0:
            r += (self.reward_range[1]-100)*(1 - (d/self._params["r_radius"])**2 )
            
        return r
    
##########################################                 
        

    def render(self, fp = None):
            #plt.clf()
        if self.params_["render"]:
            plt.cla()
            #plt.grid()
            size = 200/self.dim_[0]
            # Render the environment to the screen
            t_map = (self.map_)
            plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
            for i in range(self.dim_[0]):
                    for j in range(self.dim_[1]):
                        arr = self.act_2_dir(self.wind_[i][j])
                        plt.arrow(i,j,arr[0]/3, arr[1]/3)
            if self.agent_[0] != self.goal_[0] or self.agent_[1] != self.goal_[1]:
                plt.plot(self.agent_[0], self.agent_[1],
                        'bo', markersize=size)  # blue agent
                temp = self.act_2_dir(int(self.agent_[2]))
                plt.arrow(int(self.agent_[0]),int(self.agent_[1]),temp[0],temp[1])
                plt.plot(self.goal_[0], self.goal_[1],
                        'gX', markersize=size)
            else:
                plt.plot(self.goal_[0], self.goal_[1],
                        'bX', markersize=size) # agent and goal

            # ticks = np.arange(-0.5, self.dim_[0]-0.5, 1)
            # self.ax_.set_xticks(ticks)
            # self.ax_.set_yticks(ticks)
            # plt.xticks(color='w')
            # plt.yticks(color='w')
            plt.show(block=False)
            if fp != None:
                self.fig_.savefig(fp +"%d.png" % self.img_num_)
                self.fig_.savefig(fp +"%d.eps" % self.img_num_)
                self.img_num_ += 1
            plt.pause(1)
            #plt.close() 
        elif self.params_["print"]:
            print(self.agent_)
    
