Introduction
============

``irl-gym`` is a custom `Farama Gymansium <https://gymnasium.farama.org/>`_ derivative to be used for primarily for research purposes.
This package is intended to be used primarily with our `Decision Making toolbox <https://github.com/wvu-irl/decision-making>`_ (to be made public pending minor revisions).


Install
*******

From ``irl-gym/`` enter ``pip install -e .``

To uninstall

``pip uninstall irl_gym``


Assumptions and Specifications
******************************

Our enviroments assumes the following:


Input
-----

To declare an algorithm, use 
``env = gym.make("irl_gym/GridTunnel-v0", max_episode_steps=<desired max steps>, seed=<seed>, params=param)``

As seen above, each algorithm should accepts as input a dictionary named ``params`` containing at a minimum 
(this is not enforced but necessary to work with our algorithms).

:param r_range: (tuple) of the reward min and reward max

For rendering, include

:param render: (str) render mode (see metadata for options), *default*: "none"
:param prefix: (string) where to save images, *default*: "<cwd>/plot"
:param save_frames: (bool) save images for gif, *default*: False

For logging, 

:param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"


Environment Members
-------------------

Variables
^^^^^^^^^

action_space (spaces.<type>)
    This is consistent with `Gym standard for spaces <https://gymnasium.farama.org/api/spaces/>`_.

    We generally use ``int`` for actions but this is not required.
_log (Logger)
    Logger for printing and logging.
metadata (dict)
    Contains "render_modes" for setting render mode.
observation_space (spaces.<type>)
    This is consistent with `Gym standard for spaces <https://gymnasium.farama.org/api/spaces/>`_.

    We generally use ``dict`` for observations but this is not required.
_params (dict)
    For storing parameter values from `params`.
_state (<observation type>)
    Member for representing state.


Functions
^^^^^^^^^

**Gym Standard**
`More info on Gym <https://gymnasium.farama.org/api/env>`_

__init__(seed : int, params : dict)
    Initializes environment (we lump most of this functionality into reset).

    *INPUT* Seed for RNG, params for environment

reset(seed : int, options : dict)
    Resets the environment state, seed, and parameters.

    **Gym assumes seed is only set in** ``__init__`` **, however, we do not make this assumption as it gives greater flexibility to the planning tools.**

    *INPUT* Seed for RNG, params for environment

    *RETURNS* Observation, Info

step(a : <action type>)
    Performs desired action and increments the environment by one timestep.

    *INPUT* Desired action

    *RETURNS* Observation, Reward, Is Done, Is Truncated, Info

_get_obs()
    Private member that retrieves an observation.

    *RETURNS* Observation

_get_info()
    Private member that retrieves desired information.

    *RETURNS* Info

reward(s,a,sp)
    Calculates reward for desired state transition.

    *INPUT* State, Action, Resulting State

    *RETURNS* Reward

render()
    Renders environment as specified by ``render_mode``.


**IRL Standard**

get_actions(s)
    Retrieves available actions from given state.

    **May not return all subsquent states if they are too large, in this case it may help to factor states**

    *INPUT* State

    *RETURNS* list of actions, list of resulting states


__init__.py
***********

To add your env to the compiler, be sure your file is located in the ``envs/`` folder and
insert the following code to your ``__init__.py``::

    register(
        id='irl_gym/<EnvName>-v0',
        entry_point='irl_gym.envs:<EnvClass>',
        max_episode_steps=100,
        reward_threshold = None,
        disable_env_checker=False,
        nondeterministic = True,
        order_enforce = True,
        autoreset = False,
        kwargs = 
        {
            "params":
            {
            }
        }
    )


Citation
--------
If you are using this in your work, please cite as::

    @misc{beard2022irl_gym,
        author = {Beard, Jared J., Butts, Ronald M. , Gu, Yu},
        title = {IRL-Gym: Custom Gym environments for academic research},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/wvu-irl/irl-gym}},
    }
