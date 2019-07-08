'''
	functions
		1st. 4 frames mixing
		2nd. annealing epsilon calculation
		3rd. get_action 
'''
from parameters import *
import numpy as np

def get_epsilon(steps):
	epsilon = steps*(fin_epsilon-initial_epsilon)/fin_exploration_frame+1
	if epsilon < fin_epsilon:
		epsilon = fin_epsilon
	return epsilon
	
def get_action(epsilon,Q_val):
	if np.random.rand(1)[0] > epsilon:
		return np.argmax(Q_val)[0]
	else:
		return np.random.randint(len(Q_val))