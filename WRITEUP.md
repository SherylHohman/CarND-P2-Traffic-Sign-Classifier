#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./data_plotted_image_distribution_amongst_classes.png "Visualization_1: Distribution of Images between classes, and across Training, Validation, and Test Sets"
[image80]: ./sample_traffic_signs_from_training_set.png "Visualization_2: Sample Images from the Training Set"
[image5]: ./sample_grayscale-1channel_conversion.png "sample image converted to 1D grayscale"
[image6]: ./sample_grayscale_conversions_single_channel.png "sample images converted to 1D grayscale" 
[image7]: ./sample_grayscale_1D_to_3D_conversion.png "grayscale image converted from 1D to 3D loose tonal benefits gained in the rgb to gray conversion"
[image12]: ./sample_traffic_signs_from_training_set.png "Visualization 2"
[image10]: ./examples/grayscale.jpg "Grayscaling"
[image2]: ./examples/random_noise.jpg "Random Noise"
[image9]: ./examples/placeholder.png "Traffic Sign 1"
[image13]: ./examples/placeholder.png "Traffic Sign 2"
[image14]: ./examples/placeholder.png "Traffic Sign 3"
[image1]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SherylHohman/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set 

**In [2]** uses _numpy_ library togather basic statistics about the dataset 

I also used a [routine](http://stackoverflow.com/a/850962/5411817) to read the number of output classes directly from the signnames.csv file.

```
Number of training examples   = 34799  
Number of validation examples = 4410  
Number of testing examples    = 12630  
Image data shape = (32, 32, 3)  
Number of classes = 43  
```

####2. Include an exploratory visualization of the dataset 

**In [3]**  and **In [80]** visualizally explore the data set.  

  
Cell **In [3]** displays a bar chart showing distribution of the data  
- across the Training, Validation, and Test sets, and  
- across the classes we are training on. 

![alt text][image3]

As you can see, the distribution of each type of image is similar across the three datasets.  
This is Good.  

It can also be seen that some image classes are highly favored, while others have relatively few examples.  
- Was this favortism well chosen ?   
- Were the "most important" signs the ones with the greatest representation ?    
- How do we define "most important" ?   
  -- most common?  
  -- signs that entail the greatest danger?  
  -- signs that have the greatest consequences if misread?   
  -- signs most likely to be misread? (and how could we know the answer to this question _before_ training?)  

__

Cell **In [80]** displays some sample images from the Training set.  

![alt text][image80]

From this we can see that the exposure of images varies widely.  
It appears (from this small selection) that the images are cropped close, and are taken straight on.  
I also have a better idea of the resolution and size of the images that the network is training with.  
Seeing these images was a good insight for me.

###Design and Test a Model Architecture

####1. Identify where in your code, and Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

### PreProcessing

Cells **In [5]**, **In [6]**, **In [7]**, **In [8]**, **In [9]**, **In [73]** contain various routines for pre-processing images.   Mostly they focus on grayscaling, and mean-centering the distribution of pixel values.  

####**Grayscaling** 
Grayscaling images did not prove useful for me in my training.
In order for my LeNet architecture to work on grayscale images, they had to be represented as color images (3-channel grayscale images).  
Turning images into a single channel grayscale seemed to improve the exposure/contrast on the couple sample images I looked at. 
However once I converted it to a 3-channel rgb grayscale, the clarity returned to muddiness.  I tried various methods of scaling the pixel values across the 3 channels in attempt of retaining the visual gains I'd acheived during my single-channel grayscale conversion, but was not successful. 
Some of the attempts I made are noted in the notebook. 
When I trained on rgb grayscale images, the results were nil. 

**In [5]** `get_grayscale_datasets_1channel()` and  
**In [6]** `get_grayscale_datasets()`  

![alt text][image5]  
one channel grayscale images  
![alt text][image6]

**In [7]** `transform_grayscale_into_3D_grayscale()`   

3-channel grayscale images  
![alt text][image7]  
Notice: The result of transforming a 1D grayscale image into a 3-channel grayscale image,  
is that the contrast(?) gains from converting color to gray  are lost, even though the scale from normalization has been retained.  
![alt text][image80]  

####**Normalizing**  
#####..what did NOT work
**In  [8]** `get_per_channel_mean_zero_centered_datasets()`  
**In [84]** `get_normalized_images()`  
I tried normalizing the pixel values across the entire training dataset, and I also tried normalizing per channel.  Both methods were unsuccessful.  Although I these were the most common normalization techniques I read about, when I trained on these preprocessing techniques, the results were miserable - looking about the same as an untrained network.  
Perhaps I implemented them incorrectly.  
In retrospect, I believe the use case for that technique is for comparing, say frames from a security camera.  In that case the exposure for images is the same from frame to frame (changes with time of day).  What is different is brightness/contrast etc in different parts of the image.  
In our case, the images are taken from many different exposures, lighting conditions, color casts, etc. They are taken in different physical locations. So in this way, there would not be a uniformity across all images in the training set that we should try to normalize on.  Instead, the images are zoomed in, and while they may contain shadows cutting across an image that could "confuse" the network, generally they are fairly uniform within an image. And shadows, etc are features that we want to train on anyway, as they are going to occur "in the wild".  We want our network to recognize a sign whether it has a shadow cutting across it or not.  
This is my reasoning why the per channel, and per training set normalization techniques did not work.  

####..And why my per-image normalization technique DID work.  

**In [9]** `get_per_image_mean_centered_datasets()`  
This the normalization / preprocessing function I used to train my network.

I took each image, summed up all the pixel values in all the channels, divided this by all the pixels (32 x 32 x 3 = 3072) to arrive at the average value for a pixel in that image.

Then I subtracted this number from every pixel in the image.  This gave me "pixel" values from -128 to +128, centered at 0. _(one of those 128s should probably be 127, like -127 to 128 ??)_  

After that, I Divide every "pixel" value by the average to give me values for ranging from -1 to 1, normalized at 0.  

Though it seems more common to use 0 to 1 for images, I liked the idea of centering the image's mean at zero, the most common range/centering, it seems, for non-image data. 

I did NOT divide by standard deviation to "standardize" the set.  Although this is common, and often recommended, I found other literature indicating it was not necessarily useful for these images.  

This was the first model that I had success with.   
When it came time to upgrade my model, I wanted to change _one thing at a time_ so I could see the effects it had on my network.  
While I considered adding standardization, it also stood to reason that I would gain more by adding other techniques such as adding a dropout layer, or augmenting my data.  
This per-image mean centering Leaped ahead of other preprocessing techniques I'd tried, and was thrilled.  
I surpassed teh 93% minimum required, using this technique alone.  
I don't think (though it might be nice to try a comparison) it would have improved my results Greatly.

It was much more interesting to me to focus on my training architecture than to fiddle with a preprocessing technique that worked well :-)

#####No images to display here.  
In normalizing my data, "pixel values" were turned into values between -1 and 1.  
 
Hence the arrays representing my images'pixel values are no longer in the 0-255 range for displaying images.  
Perhaps there is a library function that can display such values as an image. I did not see one.  And decided it unnecessary to try viewing the resulting transformation visually. ;-)  



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 