# Human_level_control_through_deep_reinforcement_learning
# parameters
batch_size = 32
memory_buffer_size = 1000000
history_length = 4
target_update_frequency = 10000
gamma = 0.99
action_repeat = 4
update_frequency = 4
learning_rate = 2.5e-4
momentum = 0.95
initial_epsilon = 1
fin_epsilon = 0.1
fin_exploration_frame = 1000000
replay_start_size = 50000
no_op_max = 30

max_episodes = 500000
load_model = False # True: loading existing neuralnetwork model, False: Newly training begins!
save_frequency = 1000
save_model_path = './weight_model'
save_movie_path = './episode_movie'
