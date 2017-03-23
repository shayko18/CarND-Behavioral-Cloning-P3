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

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline i used for training and validating the model, and it contains comments to explain how the code works.
The model.py is divided to four parts:

* Part A: Help Functions. Helper function we will use during our work
* Part B: Configuration. Hyperparametres, tunable parameters and other configurations are set here
* Part C:  Preparing the data. Here we will read the data from the csv file and prepare it to enter the CNN
* Part D:Running the Model. We will run our CNN, see it's performance on the training and validation samples. We will save the model to 'model.h5' 


###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model was taken from the Nvidia paper. 
* Lines 302-303: First we have a normalization layer. In this layer we first crop the image from the top (about 40%: 65 pixels) and from the bottom (~16%: 25 pixels). After that we normalize the RGB pixels value to be between [-1 to 1) by dividing them by 127.5 and subtracting 1.0
* Lines 306-308: first three layers (1-3) are 5x5 conv2D layers. activation is 
*  (that insert non-linearity), and we subsample by 2x2
* Lines 311-312: next two layers (4-5) are 3x3 conv2D layers. activation is elu funtion which preformed better than relu
* Lines 315-321: next four layers (6-9) are fully connected layers with some dropout to reduce overfitting 
* Lines 324: we try to minimize the mse, and we are using the 'adam' optimizer
* Lines 327-329: we run the model. we use the generator we built so we don't have to store all the images in the memory. Here we use the hyper parameter that set the number of epochs we run
Later we will see a more elaborate description of the CNN 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (lines 327-328). 
We acomplished this by splitting the csv file to two portion: (line 221) 

- 80% for the training: Number of Original Training Points = 6428
- 20% for the validation: Number of Validation Points = 1608

(Later we will talk about using mode data for the training set)

In Lines 317, 319 we added a dropout layers in order to reduce overfitting. First dropout is of 50% and the seconed is of 20% 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 324).
We used betch size of 128. seems to run fast enough
We used 5 epochs to train the model. It seems to be enough
The correction factor we used for the left and right cameras was 0.25. We will explain about it later in this writeup

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
For the training data we used:

* the center, left, right cameras images. The angle on the left and right images were correctted by a tunable factor that we found after some experiments.
	* The left camera angle was correted by 0.25 
	* The right camera angle was correted by -0.25 
* we flipped each image and changed the angle on this flipped image to by the negative of the original angle.
* In my final solution I **didn't** use the option to augmnented the training data using changes in the brightness
* In my final solution I **didn't** use the option to balance the training data: use more images with low probability angles (big angles in their absolute value)

The bottom line is that for the training data we multiply the original centerred image by 6 (3 cameras and flip): Number of Training Points After Augmentation and use of all cameras = 38568

For the validation data we only used the original centerred image.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one that was shown in class. The Nvidia CNN. I thought this model might be appropriate because they used it for the same porpuse - driving a self driving car.

I wanted to increase the number of training data samples. In order to do that I used all three cameras and also fliped each image (due to the problem simmetry). I explained about it in the previus section.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I added two dropout stages is the fully connectted layers.

Then I ran the model with big number of epochs (~10) and I saw that 5 epochs will do the job because the loss stopped going down and the also the validation error was low and close to the training set error.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is how the MSE of the training and validation data changed as a function of the epoch:

![alt text][image1]

####2. Final Model Architecture

The final model architecture (model.py lines 302-321) consisted of a convolution neural network with the following layers and layer sizes: (see also previus sections)

Here is a visualization of the architecture:

![alt text][image2]


* We have a normalazation stage that containes:
	* crop: we remove irrelevant parts from the image
	* normalize: set the pixels value to be between [-1,1)
* Next we have 3 layers of 5x5 conv2D with stride of 2x2 and activation of elu that insert non linearity
* Next we have 2 layers of 3x3 conv2D with activation of elu that insert non linearity
* Next we have 4 layers of fully connected layers with dropout after the first two layers (with dropout of 0.5 and 0.2)

####3. Creation of the Training Set & Training Process

I used the data that use given to us. As I talked about before, I used all three cameras and the flipped images of those images. 
Here is an example of the image from the center, left, right and their flipped version:

![alt text][image3]

I also consider augmentation of the training data by balancing the histogram of the angles of the images. As we can see below, most of angels were close to 0.0. We can augment the low probability angles by adding more images to those bins with different brightness factor. We didn't do it. 

Here is the original histogram of the angles:

![alt text][image4]

Here is an example of the original image with the minimal, maximal and nomonal brightness factor we used:

![alt text][image5]

As I talked about before, after the collection process, I had 38568 number of data points.

