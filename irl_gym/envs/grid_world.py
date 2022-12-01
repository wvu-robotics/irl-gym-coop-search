"""
This module contains the GridworldEnv for discrete path planning

Syntax convention note: 
- Using leading "_" for function arguments (except gym standard vars)
- Using trailing "_" for member variables (except gym standard vars)
"""

__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import gym
from gym import spaces

import random
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from irl_gym.utils.utils import *


class GridWorldEnv(gym.Env):
    """   
    Simple Gridworld where agent seeks to reach goal. 
    
    ***Differs from gym standard: actions are ints and observations are dicts***
    
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
    
    - $-\,0.01 \qquad d > r_{goal} $
    
    - $1 - \dfrac{d}{r}^2 \quad \; \; d \leq r$
    
    where $d$ is the distance to the goal and $r_{goal}$ is the reward radius of the goal.
    
    **Input**
    Input parameters are passed as arguments through the ``_params`` dict.
    The corresponding keys are as follows:
    
    :param dimensions: ([x,y]) size of map, *default* [40,40]
    :param goal: ([x,y]) position of goal, *default* [10,10]
    :param state: (State) Initial state, *default*: {"pose": [20,20]}
    :param r_radius: (float) Reward radius, *default*: 5.0
    :param p: (float) probability of remaining in place, *default*: 0.1
    :param render: (bool) render on/off, *default*: False
    :param print: (bool) text render on/off, *default*: True
    :param prefix: (string) where to save images, *default*: ""
    :param save_gif: (bool) save images for gif, *default*: False
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, _params = {}):
        super(GridWorldEnv, self).__init__()
        
        self.params_ = _params
        
        if "dimensions" not in self.params_:
            self.params_["dimensions"] = [40, 40]
        if "goal" not in self.params_:
            self.params_["goal"] = [np.round(self.params_["dimensions"][0]/4), np.round(self.params_["dimensions"][1]/4)]
        if "state" not in self.params_:
            self.params_["state"]["pose"] = [np.round(self.params_["dimensions"][0]/2), np.round(self.params_["dimensions"][1]/2)]
        if "r_radius" not in self.params_:
            self.params_["r_radius"] = 5
        if "p" not in self.params_:
            self.params_["p"] = 0.1
        
        if "render" not in self.params_:
            self.params_["render"] = False
        if "print" not in self.params_:
            self.params_["print"] = False
        if self.params_["render"]:
            self.fig_ = plt.figure()
            self.ax_ = self.fig_.add_subplot(1,1,1)
            self.map_ = np.zeros(self.params_["dimensions"])
            for i in range(self.params_["dimensions"][0]):
                for j in range(self.params_["dimensions"][1]):
                    self.map_[i][j] = self.get_reward([i,j])
        if "save_gif" not in self.params_:
            self.params_["save_gif"] = False        
        if "prefix" not in _params:
            self.params_["prefix"] = current   
        if self.params_["save_gif"]:
            self.count_im_ = 0    
        
        self.a_ = [0, 1, 2, 3]
        self.action_space = gym.spaces.discrete.Discrete(4)
        self.observation_space = gym.spaces.box.Box(low =np.zeros(2), high=np.array(self.params_["dimensions"]))
    
    def reset(self, seed = None, return_info=None):
        """
        Resets environment to initial conditions/state
        
        :param seed: (int) random number generator seed, *default*: None
        :param return_info: (bool) , *default*: None
        :return: (tuple) State, reward, is_done, is_truncate, info 
        """
        super().reset(seed=seed)
        self.agent_ = self.params_["state"]
        return self.get_observation()
    
    def step(self, _action):
        """
        Increments enviroment by one timestep 
        
        :param _action: (int) random number generator seed, *default*: None
        :return: (tuple) State, reward, is_done, is_truncate, info 
        """
        _action = self.sample_transition(_action)
        self.agent_ = self.get_coordinate_move(self.agent_, _action)
        
        r = self.get_reward(self.agent_)
        if self.agent_ == self.params_["goal"]:
            done = True
        else:
            done = False
            
        return self.get_observation(), r, done, False, {}
    
    def get_observation(self):
        """
        Gets observation
        
        :return: (State)
        """
        return deepcopy({"pose": [int(self.agent_[0]), int(self.agent_[1])]})
    
    def get_reward(self, _s, a = None, _sp = None):
        """
        Gets rewards for $(s,a,s')$ transition
        
        :param _s: (State) Initial state
        :param _a: (int) Action (unused in this class), *default*: None
        :param _sp: (State) Action (unused in this class), *default*: None
        :return: (float) reward 
        """
        d = get_distance(_s["pose"], self.params_["goal"])
        if d >= self.params_["r_radius"]:
            return -0.01
        else:
            return 1 - (d/self.params_["r_radius"])**2

    def render(self, _fp = None):
        """    
        Rendering

        - blue: agent
        - green X: goal 
        - blue X: goal + agent
        """
            #plt.clf()
        # print(self.agent_)
        if self.params_["render"]:
            plt.cla()
            #plt.grid()
            size = 100/self.params_["dimensions"][0]
            # Render the environment to the screen
            t_map = (self.map_)
            print("max map ", np.max(np.max(self.map_)))
            plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
            if self.agent_[0] != self.params_["goal"][0] or self.agent_[1] != self.params_["goal"][1]:
                plt.plot(self.agent_[0], self.agent_[1],
                        'bo', markersize=size)  # blue agent
                plt.plot(self.params_["goal"][0], self.params_["goal"][1],
                        'gX', markersize=size)
            else:
                plt.plot(self.params_["goal"][0], self.params_["goal"][1],
                        'bX', markersize=size) # agent and goal

            # # ticks = np.arange(-0.5, self.params_["dimensions"][0]-0.5, 1)
            # # self.ax_.set_xticks(ticks)
            # # self.ax_.set_yticks(ticks)
            # plt.xticks(color='w')
            # plt.yticks(color='w')
            # plt.show(block=False)
            # if _fp != None:
            #     self.fig_.savefig(_fp +"%d.png" % self.img_num_)
            #     self.fig_.savefig(_fp +"%d.eps" % self.img_num_)
            #     self.img_num_ += 1
            plt.pause(1)

            plt.show()

            # plt.close() 
        elif self.params_["print"]:
            print(self.agent_)
            if self.params_["save_gif"]:
              plt.savefig(self.prefix_ + "img" + str(self.count_im_) + ".png", format="png", bbox_inches="tight", pad_inches=0.05)
              self.count_im_+=1
        

    
    def sample_transition(self, _action):
        p = self.np_random.uniform()

        if p < self.params_["p"]:
            # t = self.a_.copy()
            # t.remove(_action)
            # _action = self.np_random.choice(t)     
            _action = 4
        return _action
    

        
    def get_actions(self, _agent=None):
        n, a = self.get_neighbors(_agent["pose"])
        return a
    
    def get_neighbors(self, _position):
        neighbors = []
        neighbors_ind = []
        step = [[ 0, -1], [-1,  0], [ 0,  1], [ 1,  0]]
        for i in range(4):
            # print(_position)
            t = _position.copy()
            t[0] += step[i][0]
            t[1] += step[i][1]
            
            if t[0] >= 0 and t[1] >= 0 and t[0] < self.params_["dimensions"][0] and t[1] < self.params_["dimensions"][1]:
                neighbors.append(t)
                neighbors_ind.append(i)
        return neighbors, neighbors_ind

    def get_coordinate_move(self, _position, _action):
        _position = _position.copy()
        step = []
        if   _action == 0:
            step = [ 0, -1] # S
        elif _action == 1:
            step = [-1,  0] # W
        elif _action == 2:
            step = [ 0,  1] # N
        elif _action == 3:
            step = [ 1,  0] # E
        else:
            step = [0, 0]   #  Z

        temp = _position.copy()
        temp[0] = temp[0] + step[0]
        temp[1] = temp[1] + step[1]
        
        if temp[0] < 0:
            temp[0] = 0
        if temp[0] >= self.params_["dimensions"][0]:
            temp[0] = self.params_["dimensions"][0]-1
        if temp[1] < 0:
                temp[1] = 0
        if temp[1] >= self.params_["dimensions"][1]:
            temp[1] = self.params_["dimensions"][1]-1
        # print (_position)
        # print(temp)
        return temp

    def get_action(self, _action):
        if   _action[0] ==  0 and _action[1] == -1:
            return 0 # S
        elif _action[0] == -1 and _action[1] ==  0:
            return 1 #  W
        elif _action[0] ==  0 and _action[1] ==  1:
            return 2 # N
        elif _action[0] ==  1 and _action[1] ==  0:
            return 3 #  E
        else:
            return 4 # Z

    def write_gif(self):
        pass