"""
This module contains the GridTunnelEnv for discrete path planning with a local maxima
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
from irl_gym.envs.grid_world import GridWorldEnv


class GridTunnelEnv(GridWorldEnv):
    """   
    Simple Gridworld where agent seeks to reach goal with a local minima. 
    
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
        - $\dfrac{R_{max} - \dfrac{d}{r_{goal}}^2}{2}, \quad d \leq r_{trap}$
        - $R_{max} - \dfrac{d}{r_{goal}}^2, \quad d \leq r_{goal}$
    
        where $d$ is the distance to the goal, $r_i$ is the reward radius of the goal/trap respectively, and
        R_i$ are the reward extrema.
    
    **Input**
    
    :param seed: (int) RNG seed, *default*: None
    
    Remaining parameters are passed as arguments through the ``params`` dict.
    The corresponding keys are as follows:
    
    :param dimensions: ([x,y]) size of map, *default* [35,10]
    :param goal: ([x,y]) position of goal, *default* [10,5]
    :param state_offset: (int) distance of state from goal in +x direction, *default*: 15
    :param trap_offset: (int) distance of trap from goal in +x direction, *default*: 17
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
        super(GridTunnelEnv, self).__init__(seed = seed, params = params)

        self.reset(seed=seed, options=params)
    
    
    def reset(self, *, seed: int = None, options: dict = {}):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*:, {}
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        super().reset(seed=seed, options=options)
        self._log.debug("Reset GridWorld")

        if "state_offset" not in self._params:
            self._params["state_offset"] = 15
        if "trap_offset" not in self._params:
            self._params["trap_offset"] = 17

        self._state["pose"] = self._params["goal"].copy()
        self._state["pose"][0] += self._params["state_offset"]
        self._params["state"] = deepcopy(self._state)
        self._trap = self._params["goal"].copy()
        self._trap[0] += self._params["trap_offset"]

        self._log.info(str(self._state))
                
        return self._get_obs(), self._get_info()

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
        d2 = np.linalg.norm(sp["pose"] - self._trap)
         
        if d2 < self._params["r_radius"]:
            return self.reward_range[0] + (self.reward_range[1] - (d/self._params["r_radius"])**2)/2
        if d < self._params["r_radius"]:
            return self.reward_range[1] - (d/self._params["r_radius"])**2
        return self.reward_range[0]
    #     self.params_ = _params
        
    #     if "dimensions" in _params:
    #         self.dim_ = _params["dimensions"]
    #     else:
    #         self.dim_ = [40, 40]
            
    #     if "goal" in _params:
    #         self.goal_ = _params["goal"]
    #     else:
    #         self.goal_ = [np.round(self.dim_[0]/4), np.round(self.dim_[1]/4)]
            
    #     if "p" in _params:
    #         self.p_ = _params["p"]
    #     else:
    #         self.p_ = 0.1
            
    #     self.map_ = np.zeros(self.dim_)
        
    #     self.rng_ = None

    #     if "offset" in _params:
    #         offset = _params["offset"]
    #     else:
    #         offset = 2

    #     self.reset()
    #     self.trap_ = [self.agent_[0] + offset, self.agent_[1]]

    #     for i in range(self.dim_[0]):
    #         for j in range(self.dim_[1]):
    #             self.map_[i][j] = self.get_reward([i,j])
        
    #     self.a_ = [0, 1, 2, 3]
        
    #     if self.params_["render"]:
    #         self.fig_ = plt.figure()
    #         self.ax_ = self.fig_.add_subplot(1,1,1)
        
    #     self.rng_ = np.random.default_rng()
    #     self.count_im_ = 0
        
    #     self.action_space = gym.spaces.discrete.Discrete(4)
    #     # print(self.action_space)
    #     self.a_ = [0, 1, 2, 3]
    #     _high = np.ones(2)
    #     _high[0] = self.dim_[0]
    #     _high[1] = self.dim_[1]
    #     self.observation_space = gym.spaces.box.Box(low =np.zeros(2), high=_high)
        
    #     if self.params_["render"]:
    #         self.fig_ = plt.figure()
    #         self.ax_ = self.fig_.add_subplot(1,1,1)
            
    #     if "prefix" in _params:
    #         self.save_gif = True
    #         self.prefix_ = _params["prefix"]
    #         self.count_im_ = 0
        
    # def get_num_states(self):
    #     return self.dim_[0]*self.dim_[1]
    
    # def get_num_actions(self):
    #     return 4
    
    # def reset(self, seed = None):
    #     if seed != None:
    #         self.rng_ = np.random.default_rng(seed)
    #     elif self.rng_ == None:
    #          self.rng_ = np.random.default_rng()
             
    #     if "state" in self.params_:
    #         self.agent_ = self.params_["state"]["pose"]
    #     else:
    #         self.agent_ = [np.floor(self.dim_[0]/2), np.floor(self.dim_[1]/2)]
    #     return self.get_observation()
        
        
        
    
    # def render(self, _fp = None):
    #         #plt.clf()
    #     if self.params_["render"]:
    #         plt.cla()
    #         #plt.grid()
    #         size = 100/self.dim_[0]
    #         # Render the environment to the screen
    #         t_map = (self.map_)
    #         print("max map ", np.max(np.max(self.map_)))
    #         plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
    #         if self.agent_[0] != self.goal_[0] or self.agent_[1] != self.goal_[1]:
    #             plt.plot(self.agent_[0], self.agent_[1],
    #                     'bo', markersize=size)  # blue agent
    #             plt.plot(self.goal_[0], self.goal_[1],
    #                     'gX', markersize=size)
    #         else:
    #             plt.plot(self.goal_[0], self.goal_[1],
    #                     'bX', markersize=size) # agent and goal

    #         # ticks = np.arange(-0.5, self.dim_[0]-0.5, 1)
    #         # self.ax_.set_xticks(ticks)
    #         # self.ax_.set_yticks(ticks)
    #         plt.xticks(color='w')
    #         plt.yticks(color='w')
    #         plt.show(block=False)
    #         if _fp != None:
    #             self.fig_.savefig(_fp +"%d.png" % self.img_num_)
    #             self.fig_.savefig(_fp +"%d.eps" % self.img_num_)
    #             self.img_num_ += 1
    #         plt.pause(1)
    #         # if _fp != None:
    #         #     plt.savefig(_fp + "img" + str(self.count_im_) + ".png", format="png", bbox_inches="tight", pad_inches=0.05)
    #         #     self.count_im_+=1
    #         #plt.close() 
    #     elif self.params_["print"]:
    #         print(self.agent_)
        
    # def get_observation(self):
    #     # print({"pose": [int(self.agent_[0]), int(self.agent_[1])]})
    #     return deepcopy({"pose": [int(self.agent_[0]), int(self.agent_[1])]})
    
    
    # def get_distance(self, s1, s2):
    #     return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )
    
    
    # def get_reward(self, _s, _p = False):
    #     d = self.get_distance(_s, self.goal_)
    #     # if _p:
    #     #     print(d)
    #     if d >= 5:
    #         d = self.get_distance(_s, self.trap_)
    #         # if _p:
    #         #     print(d)
    #         #     print(_s)
    #         #     print(self.trap_)
    #         if d >= 5:
    #             return 0
    #         else:
    #             return (1 - (d**2)/25)/3
    #     else:
    #         return 1 - (d**2)/25
    
    # def sample_transition(self, _action):
    #     p = self.rng_.uniform()

    #     if p < self.p_:
    #         # t = self.a_.copy()
    #         # t.remove(_action)
    #         # _action = self.rng_.choice(t)     
    #         _action = 4
    #     return _action
    
    # def step(self, _action):
    #     # print(_action)
    #     # self.map_[int(self.agent_[0])][int(self.agent_[1])]+=1
    #     # print("------")
    #     # print(_action)
    #     _action = self.sample_transition(_action)
    #     # print(_action)
    #     self.agent_ = self.get_coordinate_move(self.agent_, _action)
        
        
    #     r = self.get_reward(self.agent_)
    #     if self.agent_ == self.goal_:
    #         done = True
    #     else:
    #         done = False
    #     # print(self.get_observation(), r, done, [])
    #     return self.get_observation(), r, done, False, {}
        
    # def get_actions(self, _agent):
    #     n, a = self.get_neighbors(_agent["pose"])
    #     return a
    
    # def get_neighbors(self, _position):
    #     neighbors = []
    #     neighbors_ind = []
    #     step = [[ 0, -1], [-1,  0], [ 0,  1], [ 1,  0]]
    #     for i in range(4):
    #         t = _position.copy()
    #         t[0] += step[i][0]
    #         t[1] += step[i][1]
            
    #         if t[0] >= 0 and t[1] >= 0 and t[0] < self.dim_[0] and t[1] < self.dim_[1]:
    #             neighbors.append(t)
    #             neighbors_ind.append(i)
    #     return neighbors, neighbors_ind

    # def get_coordinate_move(self, _position, _action):
    #     _position = _position.copy()
    #     step = []
    #     if   _action == 0:
    #         step = [ 0, -1] # S
    #     elif _action == 1:
    #         step = [-1,  0] #  W
    #     elif _action == 2:
    #         step = [ 0,  1] # N
    #     elif _action == 3:
    #         step = [ 1,  0] # NE
    #     else:
    #         step = [0, 0]   #  Z

    #     temp = _position.copy()
    #     temp[0] = temp[0] + step[0]
    #     temp[1] = temp[1] + step[1]
        
    #     if temp[0] < 0:
    #         temp[0] = 0
    #     if temp[0] >= self.dim_[0]:
    #         temp[0] = self.dim_[0]-1
    #     if temp[1] < 0:
    #             temp[1] = 0
    #     if temp[1] >= self.dim_[1]:
    #         temp[1] = self.dim_[1]-1
    #     # print (_position)
    #     # print(temp)
    #     return temp

    # def get_action(self, _action):
    #     if   _action[0] ==  0 and _action[1] == -1:
    #         return 0 # S
    #     elif _action[0] == -1 and _action[1] ==  0:
    #         return 1 #  W
    #     elif _action[0] ==  0 and _action[1] ==  1:
    #         return 2 # N
    #     elif _action[0] ==  1 and _action[1] ==  0:
    #         return 3 #  E
    #     else:
    #         return 4 # Z
        
    # def write_gif(self):
    #     pass