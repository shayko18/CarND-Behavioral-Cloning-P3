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
def get_image_angle(curr_sample, cam=0, flipped=0, corr_val=0.2, brightness_mode=0):
	corr_vec=[0.0, corr_val, 0.0-corr_val] # Correction for the steering: [center, left, right]
	filename = './data/IMG/'+curr_sample[cam].split('/')[-1]  # Mod3 will give us: center, left, right
	image = cv2.imread(filename)
	angle = float(curr_sample[3]) + corr_vec[cam]
	if flipped:
		image = cv2.flip(image,1)
		angle = 0.0-angle
	
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	if brightness_mode==1:
		image = change_brightness(image, float(curr_sample[4]))
	elif brightness_mode>1:
		image = change_brightness(image, float(brightness_mode)/1000.0)
	
	#image = cv2.resize(image , None, fx=0.5, fy=1.0, interpolation=cv2.INTER_CUBIC)
	return image, angle 


### change_brightness: Change the image brightness
###   Input: 
###		image: image in a RGB format
###		factor: the change_brightness will be multiply by this factor 
###
###   Output: 
###      image: the final image
def change_brightness(image, factor=1.0):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    #image = np.array(image, dtype = np.float64)
    image[:,:,2] = image[:,:,2]*factor
    #image[:,:,2][image[:,:,2]>255]  = 255
    #image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
	
    return image

### plot_histo: 
###   Plot histogram according to the hist and bins 
def plot_histo(hist, bins, title=''):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.figure()
    plt.bar(center, hist, align='center', width=width)
    plt.title(title)
    plt.xlabel('Angle')
    plt.ylabel('Number of appearances')
    plt.show() 
	
### generator: We use it to generate training and validation samples on the fly to the model
###   Input: 
###		samples: all the samples (lines in the csv) we have 
###		is_validation: to use for validation or training (in training we will use all cameras and also flip the images)
###		batch_size: how many samples to return each call
###		corr_val: by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras 
###		num_fliplr: augmentation factor. we use 2 for original image and the flipped one 
###
###   Output: 
###      np arrays of the images and the matching angles
def generator(samples, is_validation=False, batch_size=32, corr_val=0.2, num_fliplr=2):
	##
	## Inputs:
	##   num_fliplr: Factor of augmentation. We will augment the train samples by flipping them
	corr_vec=[0.0, corr_val, 0.0-corr_val] # Correction for the steering: [center, left, right]
	num_samples = len(samples) # Number of input samples
	num_camera = len(corr_vec) # Number of cameras we will use for the training samples
	num_total = num_samples    # Number of the total samples we will use.
	if is_validation==False:
		num_total *= (num_fliplr*num_camera)
	
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
					flipped = int(idx/num_camera)%num_fliplr
					curr_sample = samples[int(idx/(num_camera*num_fliplr))]
					brightness_mode=1
				else: # we use only the original image from the center camera
					cam = 0
					flipped = 0
					curr_sample = samples[idx]
					brightness_mode=0
				
				image,angle = get_image_angle(curr_sample, cam, flipped, corr_val, brightness_mode)
				images.append(image)
				angles.append(angle)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

			
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part B: Configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Global Params
num_fliplr  = 2    # Factor of augmentation. We will augment the train samples by flipping them
num_camera = 3    # Number of cameras we will use for the training samples

###
### Tune Params
balacing_en = True                                  # Balancing the data. Augmentation for the low probability angles
batch_size = 128                                    # The betch size for the training and validation
crop_top_ratio, crop_bottom_ratio = 0.406, 0.156    # By how many pixels to crop form the top and bottom
corr_val = 0.25                                     # by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras
low_brightness, high_brightness = 0.25, 1.25        # minimal and maximal values of brightness augmentation 

###
### Plot options
###    plot_example: Plot an example of the images at the sane time from the center, left and right
###                  Also the augmentation on those images and the cropping we will use.
###    plot_hist: Plot the angle histogram before and after balancing 
###    plot_fit: Plot the model fit (mse) for both the training and validation data as a function of the epoch 
plot_example = False  
plot_hist = False
plot_fit = False


####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part C: Preparing the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Read the csv file
print()
isHeader=True;
lines = []
all_angles = []
max_num_of_lines = -1 # put -1 to ignore
with open ('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		if isHeader:
			isHeader = False;
			continue
		lines.append(line)	
		all_angles.append(float(line[3]))
		if (max_num_of_lines!=-1 and len(lines)>max_num_of_lines):
			break


train_lines, validation_lines = train_test_split(lines, test_size=0.2)
n_train_org = len(train_lines)
n_valid = len(validation_lines)
print('Number of Original Training Points = {}'.format(n_train_org))
print('Number of Validation Points = {}'.format(n_valid))

if balacing_en:
	bins = np.arange(-11,13,2)/20.0
	hist, bins = np.histogram(np.array(all_angles), bins=bins)
	if plot_hist:
		plot_histo(hist, bins, 'Histogram of angles, Before Balancing')

	bin_minimal_val = hist.mean()#+hist.std()
	balance_per_bin = bin_minimal_val-hist
	balance_per_inst = np.ceil(balance_per_bin/hist)
	balance_per_inst = balance_per_inst.clip(min=0.0).astype(int)

	for k in range(n_train_org):
		train_lines[k][4] = 0.0
		line = train_lines[k]
		ang_bin = int(np.floor((float(line[3])-bins.min())*10.0))
		ang_bin = max(0, min(ang_bin, len(balance_per_inst)-1))
		for j in range(balance_per_inst[ang_bin]):
			line[4] = np.random.uniform(low=low_brightness, high=high_brightness)
			train_lines.append(line)
			all_angles.append(float(line[3]))

	hist, bins = np.histogram(np.array(all_angles), bins=bins)
	if plot_hist:
		plot_histo(hist, bins, 'Histogram of angles, After Balancing')

n_train = len(train_lines)*num_fliplr*num_camera
print('Number of Training Points After Augmentation and use of all cameras = {}'.format(n_train))


###
### compile and train the model using the generator function
train_generator = generator(train_lines, is_validation=False, batch_size=batch_size, corr_val=corr_val, num_fliplr=num_fliplr)
validation_generator = generator(validation_lines, is_validation=True, batch_size=batch_size)

### 
### One random image as an example and to get the image size
example_idx = random.randint(0, n_train_org-1)
example_image,example_angle = get_image_angle(train_lines[example_idx], 0, 0, corr_val)
example_X = np.array(example_image)
X_shape = np.shape(example_X)

crop_top = int(np.round((X_shape[0]*crop_top_ratio)))         # By how many pixels to crop form the top
crop_bottom = int(np.round((X_shape[0]*crop_bottom_ratio)))      # By how many pixels to crop form the bottom
print('Image Size = {}x{}x{} ; crop=[top:{} bottom:{}]'.format(X_shape[0],X_shape[1],X_shape[2],crop_top,crop_bottom))

if plot_example:
	cam_pos=['Center','left','rigth']
	is_flipped=['Orginal','Flipped']
	plt.figure(1)
	for i in range(num_camera):
		for j in range(num_fliplr):
			example_image,example_angle = get_image_angle(train_lines[example_idx], i, j, corr_val)
			example_X = np.array(example_image)
			plt.subplot(num_fliplr,num_camera,1+i+j*num_camera)
			#plt.imshow(cv2.cvtColor(example_X.squeeze(), cv2.COLOR_BGR2RGB))
			plt.imshow(example_X.squeeze())
			plt.plot([0,X_shape[1]], [crop_top,crop_top],'r')
			plt.plot([0,X_shape[1]], [X_shape[0]-crop_bottom,X_shape[0]-crop_bottom],'r')
			plt.title('idx={}, {}, {}, angle={:.3f}'.format(example_idx, cam_pos[i], is_flipped[j], example_angle))
	plt.show()
	
	brightness_example=[0.0, low_brightness, 1.0, high_brightness]
	plt.figure(2)
	for i in range(len(brightness_example)):
		plt.subplot(2,2,i+1)
		example_image,_ = get_image_angle(train_lines[example_idx], 0, 0, corr_val, int(1000*float(brightness_example[i])))
		example_X = np.array(example_image)
		plt.imshow(example_X.squeeze())		
	plt.show()

	
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part D: Running the Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Building our model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, Dropout

model = Sequential()
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(X_shape[0],X_shape[1],X_shape[2])))
model.add(Lambda(lambda x: x/127.5 - 1.0))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='elu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='elu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Conv2D(64,3,3,activation='elu'))
model.add(Conv2D(64,3,3,activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.2))
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
