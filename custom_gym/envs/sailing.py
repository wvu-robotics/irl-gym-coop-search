import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import gym
from gym import spaces

import copy
import random
import numpy as np
import matplotlib.pyplot as plt


class SailingEnv(gym.Env):
    """
    Let's an agent traverse a world starting from 0,0
    Description:
        Agent tries to get to goal. Reward decreases from 50 to 0 in radius of 5 around goal
        Wind incurs a penalty from -1 (against wind) to 0 (in wind)
        at each timestep, the wind can stay the same or rotate 45 deg with probability p.
    User defines:
        goal, transition probability
    Observations:
        agent position
    Actions:
        - L -1: turn left
        - C 0: straight
        - R 1: turn right
        -  S 0: move south        [ 0, -1]
        - SW 1: move southwest    [-1, -1]
        -  W 2: move west         [-1,  0]
        - NW 3: move northwest    [-1,  1]
        -  N 4: move north        [ 0,  1]
        - NE 5: move northeast    [ 1,  1]
        -  E 6: move east         [ 1,  0]
        - SE 7: move southeast    [ 1, -1]
    
    Transition: 
        movement 
    Rewards:
        - (-R_[0]*d) negative of distance
        -   R_[1]  goal reached
    Rendering:
        - blue: agent
        - green X: goal 
        - blue X: goal + agent
        - black: puddle
    """

    def __init__(self, _params):
        """
        Constructor, initializes state
        Args:
            _params (dict): "agent"        [x,y]    - Agent pose
                            "goal"         [x,y]    - Goal pose
                            "dim"          [x,y]    - Map dimension
                            "p"            (int)    - probability of arriving at desired state
                            "prefix" (string) - where to save images for gifs.
                            Leave unassigned if none
        Returns:
            State: State object
        """
        super(SailingEnv, self).__init__()
        self.params_ = _params
        if "dimensions" in _params:
            self.dim_ = _params["dimensions"]
        else:
            self.dim_ = [40, 40]
            
        if "goal" in _params:
            self.goal_ = _params["goal"]
        else:
            self.goal_ = [np.round(self.dim_[0]/4), np.round(self.dim_[1]/4)]
            
        if "p" in _params:
            self.p_ = _params["p"]
        else:
            self.p_ = 0.1
            
        self.map_ = np.zeros(self.dim_)
        
        self.rng_ = None
        
        self.wind_ = [-1]
        
        self.reset()
        
        self.a_ = [-1,0,1]
        
        self.fig_ = plt.figure()
        self.ax_ = self.fig_.add_subplot(1,1,1)
        
        self.action_space = gym.spaces.discrete.Discrete(3) #need to make this map to -1,0,1
        # print(self.action_space)
        # self.a_ = [0, 1, 2, 3]
        self.observation_space = gym.spaces.box.Box(3+self.dim_[0]*self.dim_[1],3+self.dim_[0]*self.dim_[1])
        # low=[0,0],high=[self.dim_])
        
        self.observation_space.high = np.ones(3+self.dim_[0]*self.dim_[1])
        self.observation_space.low = np.zeros(3+self.dim_[0]*self.dim_[1])
    
        if "prefix" in _params:
            self.save_gif = True
            self.prefix_ = _params["prefix"]
            self.count_im_ = 0

        
    def resample_wind(self):
        if len(self.wind_) != 1:
            for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    p = self.rng_.uniform()
                    if p < self.p_:
                        dir =  self.rng_.choice([-1,1])
                        dir = self.wind_[i][j] + dir
                        if dir < 0:
                            dir = 7
                        elif dir > 7:
                            dir = 0
                        self.wind_[i][j] = dir
        else:
            self.wind_ = np.zeros(self.dim_)
            for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    self.wind_[i][j] = self.rng_.choice(range(8))
        
    def get_num_states(self):
        return self.dim_[0]*self.dim_[1]
    
    def get_num_actions(self):
        return 3
    

    def reset(self, seed = None, return_info=None):
        if seed != None:
            self.rng_ = np.random.default_rng(seed)
            self.resample_wind()
            self.self.params_["state"]["wind"] = self.wind_
        elif self.rng_ == None:
            self.rng_ = np.random.default_rng()
            self.resample_wind()
            print(self.params_["state"])
            print(self.wind_)
            self.params_["state"]["wind"] = self.wind_

        if "pose" in self.params_["state"]:
            self.agent_ = self.params_["state"]["pose"]
        else:
            self.agent_ = [np.floor(self.dim_[0]/2), np.floor(self.dim_[1]/2), 0]

        self.wind_ = self.params_["state"]["wind"]
        
        return self.get_observation()

    def render(self, fp = None):
            #plt.clf()
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
        
    def get_observation(self):
        return {"pose": [int(self.agent_[0]), int(self.agent_[1]), int(self.agent_[2])], "wind": self.wind_}
    
    def get_distance(self, s1, s2):
        return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )
    
    def get_reward(self, _s, _w, _a, _sp):
        r = -0.01
        r += - self.get_distance(self.act_2_dir(_w),self.act_2_dir(_a))/(2*np.sqrt(2))
                
        d = self.get_distance(_s, self.goal_)
        d2 = self.get_distance(_sp, self.goal_)
        if d > d2:
            r += 1
            
        if _sp[0] < 10 or _sp[0] > self.dim_[0]-10 or _sp[1] < 10 or _sp[1] > self.dim_[1]-10:
            r += -0.1
        
        step = self.act_2_dir(_sp[2])
        ag = [_s[0]+step[0], _s[1]+step[1]]
        if int(ag[0]) == 0 or int(ag[0]) >= self.dim_[0] or int(ag[1]) ==0 or int(ag[1]) >= self.dim_[1]:
            r += -400
            

        if d <= 5 and d > 0:
            r += 1000*(1 - (d**2)/25)
        elif d == 0:
            r += 1100
            
        return r

    
    # def sample_transition(self, _action):
    #     p = self.rng_.uniform()

    #     if p < self.p_:
    #         t = self.a_.copy()
    #         t.remove(_action)
    #         _action = self.rng_.choice(t)     
    #     return _action
    
    def step(self, _action):
        self.map_[int(self.agent_[0])][int(self.agent_[1])]+=1
        # print("------")
        # print(self.wind_)
        wind_dir = self.wind_[int(self.agent_[0])][int(self.agent_[1])]
        # print(_action)
        # _action = self.sample_transition(_action)
        # print(_action)
        s = self.agent_.copy()
        
        self.agent_[2] += _action
        if self.agent_[2] < 0:
            self.agent_[2] = 7
        if self.agent_[2] > 7:
            self.agent_[2] = 0
        
        self.agent_ = self.get_coordinate_move(self.agent_, int(self.agent_[2]))
        
        
        r = self.get_reward(s, wind_dir, _action, self.agent_)
        if self.get_distance(self.agent_,self.goal_) < 1e-2:
            done = True
        else:
            done = False
            
        self.resample_wind()
        return self.get_observation(), r, done, {}
        
    def get_actions(self, _agent=None):

        return [-1, 0, 1]
    
    def get_neighbors(self, _position):
        neighbors = []
        neighbors_ind = []
        step = [[ 0, -1], [-1, -1], [-1,  0], [-1,  1], [ 0,  1], [ 1,  1], [ 1,  0], [ 1, -1]]
        for i in range(8):
            t = list(_position)
            t[0] += step[i][0]
            t[1] += step[i][1]
            
            if t[0] >= 0 and t[1] >= 0 and t[0] < self.dim_[0] and t[1] < self.dim_[1]:
                neighbors.append(t)
                neighbors_ind.append(i)
        return neighbors, neighbors_ind

    def act_2_dir(self, _action):
        if   _action == 0:
            return [ 0, -1] # S
        elif _action == 1:
            return [-1, -1] # SW
        elif _action == 2:
            return [-1,  0] #  W
        elif _action == 3:
            return [-1,  1] # NW
        elif _action == 4:
            return [ 0,  1] # N
        elif _action == 5:
            return [ 1,  1] # NE
        elif _action == 6:
            return [ 1,  0] #  E
        elif _action == 7:
            return [ 1, -1] # SE
        else:
            return [0, 0]   #  Z
    
    def get_coordinate_move(self, _position, _action):
        _position = _position.copy()
        step = self.act_2_dir(_action)

        temp = _position.copy()
        # print(temp)
        # print(step)
        temp[0] = temp[0] + step[0]
        temp[1] = temp[1] + step[1]
        
        if temp[0] < 0:
            temp[0] = 0
        if temp[0] >= self.dim_[0]:
            temp[0] = self.dim_[0]-1
        if temp[1] < 0:
                temp[1] = 0
        if temp[1] >= self.dim_[1]:
            temp[1] = self.dim_[1]-1
        return temp

    def get_action(self, _action):
        if   _action[0] ==  0 and _action[1] == -1:
            return 0 # S
        elif _action[0] == -1 and _action[1] == -1:
            return 1 # SW
        elif _action[0] == -1 and _action[1] ==  0:
            return 2 #  W
        elif _action[0] == -1 and _action[1] ==  1:
            return 3 # NW
        elif _action[0] ==  0 and _action[1] ==  1:
            return 4 # N
        elif _action[0] ==  1 and _action[1] ==  1:
            return 5 # NE
        elif _action[0] ==  1 and _action[1] ==  0:
            return 6 #  E
        elif _action[0] ==  1 and _action[1] == -1:
            return 7 # SE
        else:
            return 8 # Z

    def write_gif(self):
        pass