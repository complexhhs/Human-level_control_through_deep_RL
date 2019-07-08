# util module
'''
	util list:
		image_slicing
		image_resizing
		image_color_changing
		save result as movie jpg file
'''
from parameters import *
import numpy as np
import cv2
import imageio

# stack method np.stack([image_1, image_2, image_3, image_4],axis=-1)
# image pre_processing
def pre_processing(img):
	rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # true color changing
	img = img[32:-15] #image slicing
	train_image = cv2.resize(img,(84,84))
	train_image = cv2.cvtColor(train_image,cv2.COLOR_BGR2GRAY)
	return rgb_image/255., train_image/255.

# making animation_gif
def making_animation(frames,f_name):
	with imageio.get_writer('./'+str(save_movie_path)+'/'+str(f_name)+'.gif',mode='I') as writer:
		for frame in frames:
			image = imageio.imread(frame)
			writer.append_data(image)	
	# images = []
	# for frame in frames:
		# images.append(imageio.imread(frame))
	# imageio.mimsave('./'+str(f_name)+'.gif',images)

