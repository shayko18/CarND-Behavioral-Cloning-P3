#**Behavioral Cloning** 

##Writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./loss_vs_epoch.png "loss_vs_epoch"
[image2]: ./cnn.png "CNN"
[image3]: ./example_cam_and_flip.png "example cameras and flip"
[image4]: ./histo_before.png "histo before"
[image5]: ./example_brightness.png "example brightness"



---

You're reading it! and here is a link to my [project code](https://github.com/shayko18/CarND-Behavioral-Cloning-P3)
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The model.py is divided to four parts:

* Part A: Help Functions. Helper function we will use during our work. The important functions here are:
	* The generator function is implemented here to generate data for training rather than storing the training data in memory.
	* get_image_angle: read the image we want, convert the image from BGR to RGB (the format drive.py uses) and return the image and the matching angle.
* Part B: Configuration. Hyperparameters (BATCH_SIZE, N_EPOCH), tunable parameters (mainly crop ratio and angle correction for the left and right cameras) and other configurations are set here. 
* Part C:  Preparing the data. Here we will read the data from the driving_log.csv and prepare it to enter the CNN
* Part D:Running the Model. We will run our CNN, see its performance on the training and validation samples. We will save the model to 'model.h5' 


###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model was mainly based on the NVIDIA paper.

* Lines 307-308: First we have a normalization layer. In this layer we first crop the image from the top (about 40%: 65 pixels) and from the bottom (~16%: 25 pixels). After that we normalize the RGB pixels value to be between [-1 to 1) by dividing them by 127.5 and subtracting 1.0
* Lines 311-313: first three layers (1-3) are 5x5 conv2D layers. the stride is 2x2 and activation is relu
*  (that insert non-linearity).
* Lines 316-317: next two layers (4-5) are 3x3 conv2D layers. activation is relu function. 
* Lines 320-326: next four layers (6-9) are fully connected layers with some dropout (0.5 after the first FC layer and 0.2 after the second FC layer) to reduce overfitting 
* Lines 329: we try to minimize the mse, and we are using the 'adam' optimizer
* Lines 332-334: we run the model. we use the generator we built so we don't have to store all the images in the memory. Here we use the hyper parameter that set the number of epochs we run
Later we will see a more elaborate description of the CNN 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (lines 222, 244-245, 332-334). 
We accomplished this by splitting the driving_log.csv to two portion: (line 222) 

- 80% for the training: Number of Original Training Points = 6428
- 20% for the validation: Number of Validation Points = 1608

Later we will talk about using mode data for the training set and validation set (configurable)

In the model we added a dropout layers in order to reduce overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

- The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 329).
- We used batch size of 128. seems to run fast enough and give good results.
- We used 5 epochs to train the model. It seems to be enough. After that the MSE didn't changed by much.
- The correction factor we used for the left and right cameras was 0.25. We will explain about it later in this writeup
- We cropped out the not relevant parts from the image: ~40% from the top and ~15% from the bottom.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
For the training data we used:

* the center, left, right cameras images. The angle on the left and right images were corrected by a tunable factor that we found after some experiments.
	* The left camera angle was corrected by 0.25 
	* The right camera angle was corrected by -0.25 
* we flipped each image and changed the angle on this flipped image to by the negative of the original angle.
* In my final solution I **didn't** use the option to augmented the training data using changes in the brightness.
* In my final solution I **didn't** use the option to balance the training data: use more images with low probability angles (big angles in their absolute value)

The bottom line is that for the training data we multiply the original centered image by 6 (3 cameras and flip): Number of Training Points After Augmentation and use of all cameras = 38568

For the validation data we have two options. First we can use only the original centered image, because this is what the simulator will use when we test our model. The other option is to also augment the validation data the same way we did for the training data. By doing this we ensure that the validation data and the training data sees the same a-priori distribution of input samples so the MSE will be alligned between them. (we will explain it later).  

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one that was shown in class, the NVIDIA CNN. I thought this model might be appropriate because they used it for the same purpose - driving a self driving car.

I wanted to increase the number of training data samples. In order to do that I used all three cameras and also flipped each image (due to the fact that the problem of driving is symmetrical). I explained about it in the previous section.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I added some dropout stages between the layers.

Then I ran the model with big number of epochs (~10) and I saw that the validation error was constant more or less and the training error was still going down but very slowly, so I saw that 5 epochs was a good place to stop. 

As I talked about in project 2 (Traffic Sign Classifier), there is a question if we want to use in our model the a-priori distribution of the training data (In our case it is the probability for each angle). The a-priori probability could be integrate into the MSE calculation - high probability angles will be more dominated in the total MSE calculation. I decided not to use it here. The next point here is also related to this issue.
Here is how the MSE of the training and validation data changed as a function of the epochs. We plotted two validation error - one only with the image from the center camera (like the test will use) and the other also with the augmented data (3 cameras + flip). We see that because of the a-priori distribution of the angles, the error only on the center camera is lower, because it usually has small angles which has the highest a-priori probability. In order to compare the train error to the validation error we also plotted the error on validation data that uses all the augmentation - then we see what we expect - the training data error is lower than the validation error. 

![alt text][image1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


####2. Final Model Architecture

The final model architecture (model.py lines 307-326) consisted of a convolution neural network with the following layers and layer sizes: (see also previous  sections)

Here is a visualization of the architecture:

![alt text][image2]


* We have a normalization stage that contains:
	* crop: we remove irrelevant parts from the image
	* normalize: set the pixels value to be between [-1,1)
* Next we have 3 layers of 5x5 conv2D with stride of 2x2 and activation of relu that insert non linearity. 
* Next we have 2 layers of 3x3 conv2D with activation of relu that insert non linearity.
* Next we have 4 layers of fully connected layers with dropout after the first two layers. dropout was used between the layers

####3. Creation of the Training Set & Training Process

I used the data that use given to us. As I talked about before, I used all three cameras and the flipped images of those images. 
Here is an example of the image from the center, left, right and their flipped version:

![alt text][image3]

I also consider augmentation of the training data by balancing the histogram of the angles of the images. As we can see below, most of angels were close to 0.0. We can augment the low probability angles by adding more images to those bins with different brightness factor. We didn't do it since it was not necessary. Also as I talked about before we could have used the a-priori distribution in our model (so the MSE calculation will take it into account). We also didn't do that. 

Here is the original histogram of the angles:

![alt text][image4]

Here is an example of the original image with the minimal, maximal and nominal brightness factor we used:

![alt text][image5]

As I talked about before, after the collection process, I had 38568 number of data points.

###Simulation
As we can see in the video, the car meets the specifications -  No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces.

