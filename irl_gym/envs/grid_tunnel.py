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
    
        - $R_{min}, \qquad \qquad \quad \; d > r_{goal} $
        - $\dfrac{R_{max} - \dfrac{d}{r_{goal}}^2}{2}, \quad d \leq r_{trap}$
        - $R_{max} - \dfrac{d}{r_{goal}}^2, \quad \; d \leq r_{goal}$
    
        where $d$ is the distance to the goal, $r_i$ is the reward radius of the goal/trap respectively, and
        $R_i$ are the reward extrema.
    
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
        
        self._log = logging.getLogger(__name__)
        self._log.debug("Init GridTunnel")

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
        self._log.debug("Reset GridTunnel")

        if "state_offset" not in self._params:
            self._params["state_offset"] = 15
        if "trap_offset" not in self._params:
            self._params["trap_offset"] = 17

        self._state["pose"] = self._params["goal"].copy()
        self._state["pose"][0] += self._params["state_offset"]
        self._params["state"] = deepcopy(self._state)
        self._trap = self._params["goal"].copy()
        self._trap[0] += self._params["trap_offset"]

        self._log.info("Reset to state " + str(self._state))
                
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
            return self.reward_range[0] + (self.reward_range[1] - (d2/self._params["r_radius"])**2)/2
        if d < self._params["r_radius"]:
            return self.reward_range[1] - (d/self._params["r_radius"])**2
        return self.reward_range[0]
