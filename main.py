'''
	Simulation of 
		'Human level control through deep reinforcement learning' - Volodymyr Minh et al.(2015)
	simulated by Hyunseok, Whang.
	Date: 2019-07-08
	using 'gym' open source training atari breakout game
	
	Algorithm: Deep Q-learning with experience replay.
	
		Initialize replay memory D to capacity N
		Initialize action-value function Q with random weights theta
		Initialize target action-value function Q_ with weights theta_ = theta
		
		Repeat each episode
			Initialize sequence state and pre_processed sequence
			Repeat time_step
				with probabilkty epsilon select a random action a_t
				otherwise select a_t = argmax(Q(state,action))
				execute a_t, observe R and next state(game image)
				set new_state, a_t 
				store transition  (state,action, reward, new_state) in D
				sample random minibatch of transition (state,action, reward, new_state) from D
				set target
					1. reward, if episode terminated
					2. reward+gamma*max_action(Q_(new_state,action)) <- in target network, otherwise
				perform a gradient descent step on (target-Q(state,action))**2
				every C steps(change step) reset Q_ = Q
'''
from utils import *
from neural_net_buildup import *
from parameters import *
from collections import deque
import gym
import numpy as np
import random

# gym environment calling
'''
	atari-breakout action_meaning
	[None, Fire, Left, Right]
'''	
env = gym.make('Breakout-v0')

memory_buffer = deque(maxlen=memory_buffer_size)


ent_steps = 0

# main network and target network initializing
main_model = Build_up(env.action_space.n)
target_model = Build_up(env.action_space.n)

# episode begins
for episode in range(max_episodes):
	episode_step = 0
	done = False
	episode_scene = []
	observation = env.reset()
	rgb_img, train_img = pre_processing(observation)
	episode_scene.append(rgb_img)
	state = np.stack([train_img,train_img,train_img,train_img],axis=-1)
	next_state = state[:]
	
	# Deepmind method-when episode begins, an agent doesn't act anything for about maximum 30 time steps
	initial_stop = np.random.randint(no_op_max)
	for no_op in range(initial_stop):
		observation,reward,done,info = env.step(env.action_space.sample())
		rgb_img, train_img = pre_processing(obsservation)
		episode_scene.append(rgb_img)
		next_state[:-1] = next_state[1:]
		next_state[-1] = train_img[:]
		if no_op != 0:
			state = next_state[:]
		
	cur_lives = info['ale.lives']
	while not done:
		episode_step += 1
		ent_steps += 1
		
		epsilon = get_epsilon(ent_steps)
		
		if ent_steps < replay_start_size:
			act = np.random.randint(4)
		else:
			act = get_action(epsilon,
		
		# Shoulder 4th of state scene 
		state[:-1] = state[1:]
		state[-1] = train_img[:]
		
		
		