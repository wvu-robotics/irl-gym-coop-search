
import gym_examples
import gymnasium as gym
import numpy.random as r

rng = r.default_rng()
env = gym.make('gym_examples/GridWorld-v0', size=20,render_mode="human", max_episode_steps = 200)
env.reset()

is_terminated = False
done = False

while not done and not is_terminated:
    s,r,done,is_terminated,info = env.step(rng.choice(list(range(4))))
    print(info['distance'])
    if(info['distance'] > 5):
        print('aaaaaaaaaaa')
