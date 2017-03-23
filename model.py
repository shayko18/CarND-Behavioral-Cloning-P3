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

####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part A: Help Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
### get_image_angle: Retrieve an image and the cosponsoring angle (steering) from a specific line in the csv
###   Input: 
###		curr_sample: one element form the csv file
###		cam: which camera to use (0=center, 1=left, 2=right) 
###		flipped: if we want to flip (left-right) the picture and take the negative angle 
###		corr_val: by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras  
###		brightness_mode: if we want to change the brightness of the picture. 
###                      0: don't change. 1: use value pre-set value from in the "curr_sample". >1: the brightness factor will be the value/1000
###
###   Output: 
###      image: the final image. In a RGB format.
###      angle: the final angle
def get_image_angle(curr_sample, cam=0, flipped=0, corr_val=0.2, brightness_mode=0):
	corr_vec=[0.0, corr_val, 0.0-corr_val]                       # Correction for the steering: [center, left, right]
	filename = './data/IMG/'+curr_sample[cam].split('/')[-1]     # Use the desired camera
	image = cv2.imread(filename)
	angle = float(curr_sample[3]) + corr_vec[cam]
	if flipped:
		image = cv2.flip(image,1)
		angle = 0.0-angle
	
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)                # Switch to RGB format. drive.py will read in this format 
	if brightness_mode==1:                                       # change the brightness according to "brightness_mode" value
		image = change_brightness(image, float(curr_sample[-1]))  # pre-set value
	elif brightness_mode>1:
		image = change_brightness(image, float(brightness_mode)/1000.0)  # given value
	
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
    image[:,:,2] = image[:,:,2]*factor
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

	
### balancing_train_data: balancing the train data according to the angels histogram. Each "new" image will have different brightness factor
###   Input: 
###		train_lines: the original training lines from the csv file
###		all_angles: all the angels from the original csv file
###		plot_hist: lot the angle histogram before and after balancing
###
###   Output: 
###     train_lines: the new, augmented training lines
def balancing_train_data(train_lines, all_angles, plot_hist=True):
	bins = np.arange(-11,13,2)/20.0   # the bins we will use. [-0.55,-0.45,0.35...,0.55]
	hist, bins = np.histogram(np.array(all_angles), bins=bins)
	if plot_hist:
		plot_histo(hist, bins, 'Histogram of angles, Before Balancing')  # Plot histogram before balancing

	bin_minimal_val = hist.mean()                                 # minimal number of elements per bin
	balance_per_bin = bin_minimal_val-hist
	balance_per_inst = np.ceil(balance_per_bin/hist)
	balance_per_inst = balance_per_inst.clip(min=0.0).astype(int) # by how much to augmented each element per bin 

	for k in range(n_train_org):
		line = train_lines[k]
		ang_bin = int(np.floor((float(line[3])-bins.min())*10.0)) # finding the bin of the angle
		ang_bin = max(0, min(ang_bin, len(balance_per_inst)-1))
		for j in range(balance_per_inst[ang_bin]):
			line[-1] = np.random.uniform(low=low_brightness, high=high_brightness)  # set the brightness factor for each new, augmented, image
			train_lines.append(line)
			all_angles.append(float(line[3]))

	hist, bins = np.histogram(np.array(all_angles), bins=bins)
	if plot_hist:
		plot_histo(hist, bins, 'Histogram of angles, After Balancing') # Plot histogram after balancing
	return train_lines
	
	
### plot_histo: Plot histogram according to the hist and bins 
###   Input: 
###		hist: number of elements per bin
###		bins: bins value 
###		title: the string for the figure title
def plot_histo(hist, bins, title=''):
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.figure()
	plt.bar(center, hist, align='center', width=width)
	plt.title(title)
	plt.xlabel('Angle')
	plt.ylabel('Number of appearances')
	#plt.show() 
	if ((title.find('After')) < 0):
		plt.savefig('histo_before.png')
	else:
		plt.savefig('histo_after.png')
	
	
### generator: We use it to generate training and validation samples on the fly to the model
###   Input: 
###		samples: all the samples (lines in the csv) we have 
###		is_validation: to use for validation or training. 
###                    - In training we will use all cameras, the flipped images and the pre-set brightness
###                    - In validation we will use only the original center camera
###		batch_size: how many samples to return each call
###		corr_val: by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras 
###		num_fliplr: augmentation factor due to flipping. we use 2 for original image and the flipped one 
###
###   Output: 
###      np arrays of the images and the matching angles
def generator(samples, is_validation=False, batch_size=32, corr_val=0.2, num_fliplr=2):
	corr_vec=[0.0, corr_val, 0.0-corr_val] # Correction for the steering: [center, left, right]
	num_samples = len(samples)             # Number of input samples
	num_camera = len(corr_vec)             # Number of cameras we will use for the training samples
	num_total = num_samples                # Number of the total samples we will use.
	if is_validation==False:
		num_total *= (num_fliplr*num_camera)  # augmentation is done only for the training 
	
	# all the possible indexes of samples we will use.
	# In training we have: [center_samp[0], left_samp[0], right samp[0], center_samp_flip[0], left_samp_flip[0], right samp_flip[0],...center_samp[k]...]
	# In validation it is simply the original samples index and we use the center_samp
	total_idx = range(num_total)
	
	while 1: # Loop forever so the generator never terminates
		shuffle(total_idx)  # random ordering each time
		for offset in range(0, num_total, batch_size):
			batch_idx = total_idx[offset:offset+batch_size]

			images = []
			angles = []
			for idx in batch_idx:
				if is_validation==False: # Training: we use all 3 cameras, the flipped image and the pre-set brightness
					cam = (idx%num_camera)
					flipped = int(idx/num_camera)%num_fliplr
					curr_sample = samples[int(idx/(num_camera*num_fliplr))]
					brightness_mode=1  # pre-set brightness
				else:                    # Validation: we use only the original image from the center camera
					cam = 0
					flipped = 0
					curr_sample = samples[idx]
					brightness_mode=0 # don't change the brightness
				
				image,angle = get_image_angle(curr_sample, cam, flipped, corr_val, brightness_mode)
				images.append(image)
				angles.append(angle)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

		
		
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part B: Configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Global Params
num_fliplr = 2    # Factor of augmentation due to flipping. We will augment the train samples by flipping them
num_camera = 3    # Number of cameras we will use for the training samples. center, left, right

###
### Tune Params. Hyperparametres and other parameters the we can tune
BATCH_SIZE = 128                                    # The betch size for the training and validation
N_EPOCH = 5                                         # The number of epochs

balacing_en = False                                 # Balancing the data. Augmentation for the low probability angles. The augmented images will be with different brightness
crop_top_ratio, crop_bottom_ratio = 0.406, 0.156    # By what portion (ratio) we want to crop the image form the top and bottom
corr_val = 0.25                                     # by how much to "correct" the angle on the left (corr_val) or right (-corr_val) cameras
low_brightness, high_brightness = 0.25, 1.25        # minimal and maximal values of brightness augmentation. Used only if balacing_en is True

###
### Plot options
###    plot_example: Plot an example of the images at the same time from the center, left and right
###                  Also the augmentation on those images and the cropping we will use.
###    plot_hist: Plot the angle histogram before and after balancing (Only if balacing_en is True)
###    plot_fit: Plot the model fit (mse) for both the training and validation data as a function of the epoch 
plot_example = True  
plot_hist    = balacing_en
plot_fit     = True



####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part C: Preparing the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Read the csv file
###   We will have at the end:
###       - a long list (~8K) of the csv lines
###       - a long list (~8K) of just the angles (we will use it for the histogram)
print()
isHeader=True;                    # we will skip the first line - this is the header
lines = []                        # a list that will save all the csv lines data
all_angles = []                   # a list that will save all the angles we had in the csv
max_num_of_lines = -1             # for debug. if we want to limit the number of line we want to read. put -1 to ignore
with open ('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		if isHeader:              # we skip the first line - this is the header
			isHeader = False;
			continue
		line.append(0.0)          # we append a new column. We might want to use it to set the brightness factor, so we make sure we set it to zero (disable)
		lines.append(line)	
		all_angles.append(float(line[3]))
		if (max_num_of_lines!=-1 and len(lines)>max_num_of_lines):
			break


###
### Split the lines to training and validation (20%). We print the size of the original training set (before augmentation) an validation set.
train_lines, validation_lines = train_test_split(lines, test_size=0.2)
n_train_org = len(train_lines)
n_valid = len(validation_lines)
print('Number of Original Training Points = {}'.format(n_train_org))
print('Number of Validation Points = {}'.format(n_valid))

###
### If we want to balance the training set according to their angles initial histogram. 
if balacing_en:
	balancing_train_data(train_lines, all_angles, plot_hist=plot_hist)

###
### We print the size of the final training set (after augmentation)
n_train = len(train_lines)*num_fliplr*num_camera
print('Number of Training Points After Augmentation and use of all cameras = {}'.format(n_train))

###
### compile and train the model using the generator function
train_generator = generator(train_lines, is_validation=False, batch_size=BATCH_SIZE, corr_val=corr_val, num_fliplr=num_fliplr)
validation_generator = generator(validation_lines, is_validation=True, batch_size=BATCH_SIZE)

### 
### One random image as an example and to get the original image size (X_shape)
###     After we have the original image size we can calculate the number of pixels to crop from the top and bottom
###     We print the original image size and crop values.
example_idx = random.randint(0, n_train_org-1)
example_image,example_angle = get_image_angle(train_lines[example_idx], 0, 0, corr_val)
example_X = np.array(example_image)
X_shape = np.shape(example_X)                                    # original image size
crop_top = int(np.round((X_shape[0]*crop_top_ratio)))            # By how many pixels to crop form the top
crop_bottom = int(np.round((X_shape[0]*crop_bottom_ratio)))      # By how many pixels to crop form the bottom
print('Image Size = {}x{}x{} ; crop=[top:{} bottom:{}]'.format(X_shape[0],X_shape[1],X_shape[2],crop_top,crop_bottom))

###
### In the first figure we will print an example of the:
###    - original image from the center
###    - the left and right cameras 
###    - all of those images after we flip them.
###    - red lines will indicate where we will crop once we enter the CNN
### In the second figure we will print the original image and the original image in three levels of brightness: lowest value, highest and 1.0 (same brightness)
if plot_example:
	cam_pos=['Center','left','rigth']
	is_flipped=['Orginal','Flipped']
	plt.figure(figsize=(13,7))
	for i in range(num_camera):
		for j in range(num_fliplr):
			example_image,example_angle = get_image_angle(train_lines[example_idx], i, j, corr_val)
			example_X = np.array(example_image)
			plt.subplot(num_fliplr,num_camera,1+i+j*num_camera)
			plt.imshow(example_X.squeeze())
			plt.plot([0,X_shape[1]], [crop_top,crop_top],'r')
			plt.plot([0,X_shape[1]], [X_shape[0]-crop_bottom,X_shape[0]-crop_bottom],'r')
			plt.title('idx={}, {}, {}, angle={:.3f}'.format(example_idx, cam_pos[i], is_flipped[j], example_angle))
			plt.xlim([0,X_shape[1]])
	#plt.show()
	plt.savefig('example_cam_and_flip.png')

	brightness_example=[0.0, low_brightness, 1.0, high_brightness]
	plt.figure(figsize=(10,10))
	for i in range(len(brightness_example)):
		plt.subplot(2,2,i+1)
		example_image,_ = get_image_angle(train_lines[example_idx], 0, 0, corr_val, int(1000*float(brightness_example[i])))
		example_X = np.array(example_image)
		plt.imshow(example_X.squeeze())
		if i==0:
			plt.title('Original Image')	
		else:
			plt.title('brightness factor={}'.format(brightness_example[i]))		
	#plt.show()
	plt.savefig('example_brightness.png')

	
	
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part D: Running the Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### Building our model using keras. We will use Nvidia CNN that was suggested, and the generator we built so we don't have to store all the images in the memory   
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, Dropout

model = Sequential()
# normalizing stage
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(X_shape[0],X_shape[1],X_shape[2])))  # crop the image to use only relevant information 
model.add(Lambda(lambda x: x/127.5 - 1.0))                         # normalize the pixels value around 0 [-1 to 1]

# first three 5x5 conv layers. the striide is 2x2 and the activation is relu (that insert non-linearity)   
model.add(Conv2D(24,5,5,subsample=(2,2),activation='elu'))    # output is 33x158x24
model.add(Conv2D(36,5,5,subsample=(2,2),activation='elu'))    # output is 15x77x36
model.add(Conv2D(48,5,5,subsample=(2,2),activation='elu'))    # output is 6x37x48

# next two 3x3 conv layers. activation is relu
model.add(Conv2D(64,3,3,activation='elu'))  # output is 4x35x64
model.add(Conv2D(64,3,3,activation='elu'))  # output is 2x33x64

# next layers are fully connected with some dropout to reduce overfitting 
model.add(Flatten())      # output is 1x4224     
model.add(Dense(100))     # output is 1x100
model.add(Dropout(0.5))
model.add(Dense(50))      # output is 1x50
model.add(Dropout(0.2))   
model.add(Dense(10))      # output is 1x10
model.add(Dense(1))       # final estimated value from the CNN

# we use the mse and the default parameters for the "adam" optimizer algorithm
model.compile(loss='mse', optimizer='adam')   

# we use the generator we built so we don't have to store all the images in the memory             
history_object = model.fit_generator(train_generator, samples_per_epoch=n_train, \
            validation_data=validation_generator, nb_val_samples=n_valid, \
            nb_epoch=N_EPOCH) # The number of epochs

###
### plot the training loss and the validation loss as a function of the epoch
if plot_fit:
	plt.figure()
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	#plt.show()
	plt.savefig('loss_vs_epoch.png')

###
### Save the model
model.save('model.h5')
print('Model Saved')

