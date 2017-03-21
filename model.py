import os
import csv
import cv2
import numpy as np
from scipy import ndimage
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt

####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part A: Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
### get_image_angle: Retrieve a image and a angle (steering) from a specific line from the csv
###   Input: 
###		curr_sample: one element form the csv file
###		cam: which camera to use (0=center, 1=left, 2=right) 
###		flipped: if we want to flip (left-right) the picture and take the negative angle 
###		corr_val: by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras  
###
###   Output: 
###      image: the final image
###      angle: the final angle
def get_image_angle(curr_sample, cam=0, flipped=0, corr_val=0.2):
	corr_vec=[0.0, corr_val, 0.0-corr_val] # Correction for the steering: [center, left, right]
	filename = './data/data/IMG/'+curr_sample[cam].split('/')[-1]  # Mod3 will give us: center, left, right
	image = cv2.imread(filename)
	angle = float(curr_sample[3]) + corr_vec[cam]
	if flipped:
		image = cv2.flip(image,1)
		angle = 0.0-angle

	return image, angle 

	
### generator: We use it to generate training and validation samples on the fly to the model
###   Input: 
###		samples: all the samples (lines in the csv) we have 
###		is_validation: to use for validation or training (in training we will use all cameras and also flip the images)
###		batch_size: how many samples to return each call
###		corr_val: by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras 
###		num_augmn: augmentation factor. we use 2 for original image and the flipped one 
###
###   Output: 
###      np arrays of the images and the matching angles
def generator(samples, is_validation=False, batch_size=32, corr_val=0.2, num_augmn=2):
	##
	## Inputs:
	##   num_augmn: Factor of augmentation. We will augment the train samples by flipping them
	corr_vec=[0.0, corr_val, 0.0-corr_val] # Correction for the steering: [center, left, right]
	num_samples = len(samples) # Number of input samples
	num_camera = len(corr_vec) # Number of cameras we will use for the training samples
	num_total = num_samples    # Number of the total samples we will use.
	if is_validation==False:
		num_total *= (num_augmn*num_camera)
	
	#
	# all the possible indexes of samples we will use.
	# In training we have: [center_samp[0], left_samp[0], right samp[0], center_samp_flip[0], left_samp_flip[0], right samp_flip[0],...center_samp[k]...]
	# In validation it is simply the original samples index and we use the center_samp
	total_idx = range(num_total)
	
	while 1: # Loop forever so the generator never terminates
		shuffle(total_idx)
		for offset in range(0, num_total, batch_size):
			batch_idx = total_idx[offset:offset+batch_size]

			images = []
			angles = []
			for idx in batch_idx:
				if is_validation==False: # we use all 3 cameras and the flipped image
					cam = (idx%num_camera)
					flipped = int(idx/num_camera)%num_augmn
					curr_sample = samples[int(idx/(num_camera*num_augmn))]
				else: # we use only the original image from the center camera
					cam = 0
					flipped = 0
					curr_sample = samples[idx]
				
				image,angle = get_image_angle(curr_sample, cam, flipped, corr_val)
				images.append(image)
				angles.append(angle)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

			
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part B: Configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Global Params
num_augmn  = 2    # Factor of augmentation. We will augment the train samples by flipping them
num_camera = 3    # Number of cameras we will use for the training samples

###
### Tune Params
crop_top, crop_bottom = 65, 25 # By how many pixels to crop form the top and bottom
corr_val = 0.2                 # by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras

###
### Plot options
###    plot_example: Plot an example of the images at the sane time from the center, left and right
###                  Also the augmentation on those images and the cropping we will use.
###    plot_fit: Plot the model fit (mse) for both the training and validation data as a function of the epoch 
plot_example = False  
plot_fit = False


####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part C: Preparing the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Read the csv file
print()
isHeader=True;
lines = []
max_num_of_lines = -1 # put -1 to ignore
with open ('./data/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		if isHeader:
			isHeader = False;
			continue
		lines.append(line)	
		if (max_num_of_lines!=-1 and len(lines)>max_num_of_lines):
			break
		
train_lines, validation_lines = train_test_split(lines, test_size=0.2)
n_train_org = len(train_lines)
n_train = len(train_lines)*num_augmn*num_camera
n_valid = len(validation_lines)
print('Number of Original Training Points = {}'.format(n_train_org))
print('Number of Training Points After Augmentation and use of all cameras = {}'.format(n_train))
print('Number of Validation Points = {}'.format(n_valid))

###
### compile and train the model using the generator function
train_generator = generator(train_lines, is_validation=False, batch_size=32, corr_val=corr_val, num_augmn=num_augmn)
validation_generator = generator(validation_lines, is_validation=True, batch_size=32)

### 
### One random image as an example and to get the image size
example_idx = random.randint(0, n_train_org-1)
example_image,example_angle = get_image_angle(train_lines[example_idx], 0, 0, corr_val)
example_X = np.array(example_image)
X_shape = np.shape(example_X)
print('Image Size = {}x{}x{}'.format(X_shape[0],X_shape[1],X_shape[2]))

if plot_example:
	cam_pos=['Center','left','rigth']
	is_flipped=['Orginal','Flipped']
	plt.figure(1)
	for i in range(num_camera):
		for j in range(num_augmn):
			example_image,example_angle = get_image_angle(train_lines[example_idx], i, j, corr_val)
			example_X = np.array(example_image)
			plt.subplot(num_augmn,num_camera,1+i+j*num_camera)
			plt.imshow(cv2.cvtColor(example_X.squeeze(), cv2.COLOR_BGR2RGB))
			plt.plot([0,X_shape[1]], [crop_top,crop_top],'r')
			plt.plot([0,X_shape[1]], [X_shape[0]-crop_bottom,X_shape[0]-crop_bottom],'r')
			plt.title('idx={}, {}, {}, angle={:.3f}'.format(example_idx, cam_pos[i], is_flipped[j], example_angle))
	plt.show()

	
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part D: Running the Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Building our model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D

model = Sequential()
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(X_shape[0],X_shape[1],X_shape[2])))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=n_train, \
            validation_data=validation_generator, nb_val_samples=n_valid, \
            nb_epoch=3)

if plot_fit:
	plt.figure(2)
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

###
### Save the model
model.save('model.h5')
print('Saved Model')

exit()
