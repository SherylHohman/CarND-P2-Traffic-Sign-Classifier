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


Set 1, cell 39 `traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames`  

[image3901]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_14-3.5-stop-1.jpg 
"14: Stop, Traffic Signs from the Web, Set 1: image 1"  

[image3902]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_14-3.5-stop-2.jpg 
"14: Stop, Traffic Signs from the Web, Set 1: image 2"  

[image3903]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_17-no-entry-1-bw.jpg 
"17: No Entry, Traffic Signs from the Web, Set 1: image 3"  

[image3904]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_18-general-caution-1.jpg 
"18: General Caution, Traffic Signs from the Web, Set 1: image 4"

[image3905]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_18-general-caution-2.jpg 
"18: General Caution, Traffic Signs from the Web, Set 1: image 5"

[image3906]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_18-general-caution-3.jpg 
"18: General Caution, Traffic Signs from the Web, Set 1: image 6"  

[image3907]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_34-left-turn-ahead-1.jpg 
"34: Left Turn Ahead, Traffic Signs from the Web, Set 1: image 7"  

[image3908]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_42_no_overtaking-by-lorries-1.jpg 
"42: No Overtaking Lorries, Traffic Signs from the Web, Set 1:  image 8"  

[image3909]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_7-speed_limit_100km-1.jpg 
"7: Speed Limit 100km, Traffic Signs from the Web, Set 1: image 9"  

[image3910]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_7-speed_limit_100km-2.jpg 
"7: Speed Limit 100km, Traffic Signs from the Web, Set 1: image 10"  

Set 1, cell 39 `traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames`
_14-3.5-stop-1.jpg
_14-3.5-stop-2.jpg
_17-no-entry-1-bw.jpg
_18-general-caution-1.jpg
_18-general-caution-2.jpg
_18-general-caution-3.jpg
_34-left-turn-ahead-1.jpg
_42_no_overtaking-by-lorries-1.jpg
_7-speed_limit_100km-1.jpg
_7-speed_limit_100km-2.jpg

------------------------------------------------------------------------------

Set 2, cell 38: `traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames`  

[image3801]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/B_42_7-100kmspeedLimit_AND_no_overtaking-by-lorries-2.jpg
"Both 42, 7: 100km Speed Limit, and No Overtaking by Lorries, Traffic Signs from the Web, Set 2: image 1"  

[image3802]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/N_0_2-speed_limit_25kmh-1.jpg
"N/A 0 or 2, Speed Limit 25km, Traffic Signs from the Web, Set 2: image 2"  

[image3803]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/N_1-30km_Minimum.jpg
"N/A 1, Minimum Speed 30km, Traffic Signs from the Web, Set 2: image 3"  

[image3805]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/N_3_6_32-end-of-speed-limit-60km-1.jpg
"N/A 3 or 6 or 32, End of 60km Speed Limit, Traffic Signs from the Web, Set 2: image 5"

[image3806]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/N_5-recommended-speed-80-obsolete-1.jpg
"N/A 5, Recommended Speed 80 (obsolete), Traffic Signs from the Web, Set 2: image 6"  

[image3807]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/N_6_5_32-end-of-80km-speed-obsolete-1.jpg
"N/A 6 or 5 or 32, End of 80km Speed Limit, Traffic Signs from the Web, Set 2: image 7"  



Set 2, cell 38: `traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames`
B_42_7-100kmspeedLimit_AND_no_overtaking-by-lorries-2.jpg
N_0_2-speed_limit_25kmh-1.jpg
N_1-30km_Minimum.jpg
N_3_6_32-end-of-speed-limit-60km-1.jpg
N_5-recommended-speed-80-obsolete-1.jpg
N_6_5_32-end-of-80km-speed-obsolete-1.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SherylHohman/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set 

**In [106]** uses _numpy_ library togather basic statistics about the dataset 

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

**In [5]**, **In [6]**, **In [7]**, **In [8]**, **In [9]**, **In [73]**  
Contain various routines for pre-processing images.   Mostly they focus on various   
ways of grayscaling, and mean-centering the distribution of pixel values.  

####**Grayscaling**  
#####**..what did NOT work**  
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
#####**..what did NOT work**   
**In  [8]** `get_per_channel_mean_zero_centered_datasets()`   
**In [84]** `get_normalized_images()`   
I tried normalizing the pixel values across the entire training dataset, and I also tried normalizing per channel.  Both methods were unsuccessful.  Although I these were the most common normalization techniques I read about, when I trained on these preprocessing techniques, the results were miserable - looking about the same as an untrained network.  
Perhaps I implemented them incorrectly.  
In retrospect, I believe the use case for that technique is for comparing, say frames from a security camera.  In that case the exposure for images is the same from frame to frame (changes with time of day).  What is different is brightness/contrast etc in different parts of the image.  
In our case, the images are taken from many different exposures, lighting conditions, color casts, etc. They are taken in different physical locations. So in this way, there would not be a uniformity across all images in the training set that we should try to normalize on.  Instead, the images are zoomed in, and while they may contain shadows cutting across an image that could "confuse" the network, generally they are fairly uniform within an image. And shadows, etc are features that we want to train on anyway, as they are going to occur "in the wild".  We want our network to recognize a sign whether it has a shadow cutting across it or not.  
This is my reasoning why the per channel, and per training set normalization techniques did not work.  

####**..And why my per-image Normalization technique DID work.**   

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

#####**No images to display here.**  
In normalizing my data, "pixel values" were turned into values between -1 and 1.  
  
Hence the arrays representing my images'pixel values are no longer in the 0-255 range for displaying images.  
Perhaps there is a library function that can display such values as an image. I did not see one.  And decided it unnecessary to try viewing the resulting transformation visually. ;-)  


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)  

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

**In [1]** Reads in the image data.  The data was pre-separated into Training, Validation, and Testing sets.  
```  
Number of classes = 43  
Number of Training examples   = 34799 : 67.1% of total   
Number of Validation examples =  4410 :  8.5% of total  
Number of Testing examples    = 12630 : 24.4% of total  
```  
I might have designed the Validation and Testing sets to be of about equal size, at about  a  
70/15/15 or a 60/20/20 split, based on reading a few references to those being good generic splits.   
We are using Validation as part of the training, perhaps that matters.  
I was Very happy with the distribution across sets.!!  
I also figure the Ya'all knew what you were doing when you handed us pre-split data !! :-)  
![alt text][image3]  

**In [3]** shows that the distribution of images for each class was relatively uniformly distributed across the three sets.  This looked like a great split to me.  

**In []** Shuffles the Training and Validation image sets.  No reason to shuffle the Test set.  (The training images are reshuffled prior to each batch) **(TODO: Verify)**) And the Verification images are shuffled prior to evaluation at the end of each Epoch.  

**In [76]** Chooses a Preprocessing alogrithm to use on the data.  
I obtained good results from the **In [9]** `get_per_image_mean_centered_datasets()` alogrithm.  
It also sets the `learning_rate`, `sigma`  

**In [10]** Sets some "Constants" for my training alogrithm.  
```   
EPOCHS = 100  
BATCH_SIZE = 128  

padding = "VALID"   
stride = 1  
strides = [1, stride, stride, 1]  
pool_stride = 2  
pool_strides = [1, pool_stride, pool_stride, 1]  
ksize = pool_strides  
```   
I also defined functions to calculate/return:  
`filter_size(in_size, out_size, stride)`  , and  
`output_size(in_size, filter_size, stride)`  
for my convolution layers using 'VALID' padding  

##### Data Augmentation **TODO: FIX**  
Although I would really like to add a function for augmenting my data, I did not get that far (yet).  
I would probably begin by rotating all images in a batch by some randomly chosen constant value. Perhaps in the range of +- 15 or 20 degrees.  
Zoom might be another method.  Adding a color cast, lightening or darkening all pixels, or turning an entire batch into grayscale images (randomly choose some percentage of batches to be converted).  
Any of these methods would simulate having additional training data to feed through my network.  The network would take longer to train, and would be even less prone to overfitting.  It would also likely be more accurate.  

##### Dropout  **TODO: FIX**  
I chose to use Dropouts to address the overfitting issue I ran into with my first Good training Model.  
I added a dropout to the end of my first and second fully connected layers (Layer 3, and Layer 4).  
The use of dropouts, handled overfitting quite well.  
Dropout decreased the Loss in my validation set.  

It also took longer to train, and increased the frequency of oscillation, though it decreased the Magnitued of oscillation <loss and accuracy>.  I prefer this steadier average value.  

##### Overfitting  **TODO: FIX** 
My first successful model, while it worked quite well (achieved 95%), I was not happy with the training curves obtained during training.  
Specifically, while Training and Validation Loss quickly decreased initially, after some time, Training Loss continued to drop until it was down to 0% (and 100% accuracy.)  Meanwhile, the Validation loss began a slow climb.  Validation Accuracy Oscillated, but remained between (**TODO**) 90 and 95.  
Since Training reached a perfect fit, while Validation loss continued to slowly climb, I concluded that my model was overfitting to the training data.  

The goal of my Second Model was to address overfitting.  I wanted to change only 1 element at a time, so I could evaluate the affect that technique alone had on training my model.  

I chose to use Dropouts to address the overfitting issue I ran into with my first Good training Model.  
The use of dropouts, handled overfitting quite well.  

Another technique I could have chosen to adress the overfitting I saw in my first successfully trained model, would have been to use Data Augmentation.  
While this is a technique I would like to try, I did not do so on this model.


With the use of dropouts, I do not need Data Augmentation to address overfitting.  
However Data Augmentation would likely improve classification accuracy (and decrease loss).  
In the images I tested my trained network on, I included grayscale images, signs that were taken at an angle, and images that had poor croppings.  For this reason, I believe an augmented dataset would improve Classification, and Confidence results on the found images I supplied.  

#### Architecture  

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.  

12, 13, 75

#### LeNet Architecture
**In [75]** is LeNet architecture  
**In [12]**, **In[13]** contain implementation details for conv and fcc layers  

```  
Layer                     In-Shape      Out-Shape   Description / Settings    Function Used  

Input Layer                           (32, 32, 3)  RBG Image, Preprocessed  

Layer 1:  
convolution             (32, 32, 3)   (28, 28, 6)   filter = ?  stride = 1    tf.nn.conv2d()  
activation   RELU       (28, 28, 6)   (28, 28, 6)                             tf.nn.relu()  
pooling      Max Pool   (28, 28, 6)   (14, 14, 6)   ksize  = 2  stride = 2    tf.nn.max_pool  

Layer 2:  
convolution             (14, 14,  6)  (10, 10, 16)  filter = ?  stride = 1    tf.nn.conv2d()  
activation   RELU       (10, 10, 16)  (10, 10, 16)                            tf.nn.relu()  
pooling      Max Pool   (10, 10, 16)  ( 5,  5, 16)  ksize  = 2  stride = 2    tf.nn.max_pool()  

Flatten:                ( 5,  5, 16)  (          )  

Layer 3:  
fully connected                (120)         (120)                            prev * weights + bias  
activation    RELU             (120)         (120)                            tf.nn.relu()  
dropout                        (120)         (120)  keep_probability = 0.5    tf.nn.dropout()  
 
Layer 4:  
fully connected                (120)          (84)                            prev * weights + bias  
activation    RELU              (84)          (84)                            tf.nn.relu()  
dropout                         (84)          (84)  keep_probability = 0.5    tf.nn.dropout()  

Layer 5:  
fully connected                 (84)          (43)                            prev * weights + bias  
logits  
```  
sparce_softmax_.... (**TODO**)

 Each Layer begins with all bias elements initialized to 0 
 and weights randomly distributed along a normal distribution ??centered??(mean) at zero, and standard deviation of 0.1 ?? aka in the range of +- 0.1 ?? (**TODO**)  

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.  

(**TODO**)
The code for training the model is located in the eigth cell of the ipython notebook.  

To train the model, I used an ....  

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.  

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook. 

(**TODO**) 

My final model results were:  
* training set accuracy of ?  
* validation set accuracy of ?   
* test set accuracy of ?  

(**TODO**)
If an iterative approach was chosen:  
* What was the first architecture that was tried and why was it chosen?  
* What were some problems with the initial architecture?  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
(**TODO**)
* Which parameters were tuned? How were they adjusted and why?  
* What are some of the important design choices and why were they chosen?  For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  

If a well known architecture was chosen:  
* What architecture was chosen?  
* Why did you believe it would be relevant to the traffic sign application?  
(**TODO**)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
 

###Test a Model on New Images  

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.  

(**TODO**)

Here are five German traffic signs that I found on the web:  

![alt text][image4] ![alt text][image5] ![alt text][image6]  
![alt text][image7] ![alt text][image8]  

The first image might be difficult to classify because ...  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).  

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.  

(**TODO**)

Here are the results of the prediction:  

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...  

#### TODO:
#####  NOTE: I added additional crops of problem images to see how they would compare.
#####  I also deleted an image that was a useless poor image choice to begin with
#####  I did NOT Update My Analysis or Accuracy reporting to reflect the changes in this dataset.

#### Set_1 My classifier performed terrible on the images I expected good results on.  
Signs 1, 6: To be fair, I could have cropped these better. 

Sign 6: I expect should get a correct answer with closer cropping.  
    Then again, I was overly optimistic that it would do well even given the bad crop I handed it.  
    I can imagine that the crop I gave it lent a partial figure-8 shape to confuse it, though at the wrong scale??  
    
Sign 2: Was in black an white, so it is missing color information, which may have otherwise helped nudge it to the correct answer.  
- Additionally, the sign was skewed at an angle, and placed against another sign,  
- so it's overall shape could be difficult to discern.  
- Unfortunately, the misinterpretation is Grave:   
  - 'No Entry' sign became an "End All Speed and Passing Limits"  
  - the exact Opposite of the intended !  
        
Sign 3:  No Idea what happened there! Perhaps the edges of the triangle resembled a 5, though at the wrong scale??  
    If so this is similar to what I "imagine" threw it off for image 6. Or I just have an imagination.  
    
Sign 4 and 5, Fortunately, It got these Correct !!  

So this classifier was correct on 2/6 images, or 33% correct.  
That's Far lower than the test, and validation sets!!  It's downright Terrible.   

#### Set_2 consisted of street signs that were not part of the street sign names my classifier was trained on.  It was impossible for my classifier to get any of these correct.

- Indeed, the correct sign names do not even exist in the csv signnames file,   
- I was simply curious how it would interpret them: would see the same type similarities **I** see when looking at them?  
- Would it make the same choices I did (not knowing German signs) ?  
    
Sign 2: Surprisingly, it got almost about as close a guess as possible.   Must be beginner's luck.

Sign 5: Was also surprisingly good guess. 

Signs 3, 6, 7: These three signs I Expected the classifier to, no-brainer, return specific predictions.  
    Nope. It passed over images which I think they look Very similar to, 
    in favor of images that I think look not at all similar to these signs.   What was its "thinking" process ??  
    
Signs 1, 4: I expected rubbish responses to these rubbish images. Never-the-less, I see no resemblence to what it _did_ choose.  
    I guess NaN was not an option?  

---------------------------------------------------



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)  

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.  

(**TODO**)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...  


**In [39]**  **First set of images**


![alt text][image3901]
  99.17% Road work
   0.34% Speed limit (30km/h)
   0.26% Stop
   0.20% Yield
   0.00% No entry


![alt text][image3902]
  54.89% Speed limit (50km/h)
  21.63% Speed limit (70km/h)
  10.41% Speed limit (30km/h)
   9.42% Speed limit (80km/h)
   3.58% Stop


![alt text][image3903]
  83.94% End of all speed and passing limits
   7.62% End of no passing
   4.50% Ahead only
   1.26% Go straight or right
   0.93% End of speed limit (80km/h)


![alt text][image3904]
  62.78% Speed limit (50km/h)
  36.94% Speed limit (30km/h)
   0.18% Wild animals crossing
   0.08% Double curve
   0.00% Dangerous curve to the left


![alt text][image3905]
  99.99% General caution
   0.00% Traffic signals
   0.00% Road work
   0.00% Bumpy road
   0.00% Road narrows on the right


![alt text][image3906]
  99.93% General caution
   0.06% Traffic signals
   0.00% Road narrows on the right
   0.00% Road work
   0.00% Beware of ice/snow


![alt text][image3907]
 100.00% Turn left ahead
   0.00% Yield
   0.00% Go straight or right
   0.00% End of all speed and passing limits
   0.00% Ahead only


![alt text][image3908]
 100.00% No passing for vehicles over 3.5 metric tons
   0.00% Priority road
   0.00% End of no passing by vehicles over 3.5 metric tons
  -0.00% Speed limit (20km/h)
  -0.00% Speed limit (30km/h)


![alt text][image3909]
  76.27% Speed limit (80km/h)
  11.34% Speed limit (60km/h)
  10.05% Speed limit (50km/h)
   1.51% Stop
   0.73% No vehicles


![alt text][image3910]
  99.52% Speed limit (60km/h)
   0.47% Speed limit (80km/h)
   0.00% No passing for vehicles over 3.5 metric tons
   0.00% No passing
   0.00% Turn left ahead

---   

- Interesting how cropping changed the predictions.   
 -- Good Cropping matters. (Giving my trained model, anyhow)
 -- Stop Sign: better crop (tho not great), stop was rated 3rd on the list, but confidence was lower. worse crop, was > 10 x more confident in "stop" as a choice, though it was now 5th. Of course, this is also reflected in that the worse crop lowered it's confidence in anything overall. Better crop gave it a 99% certainty in a wrong answer, vs a 55% top certainty in the close cropped version. 
- I'm surprised that the 100km/h speed limit did not even make the list, even with good cropping.  
- How did the second to last image have 2 predictions at less than 0 % ??  
 --Is this a red-flag that something is wrong (with alogrithm) ??  Or that rounding hit an overflow ??  



--------------------------------------------------------------------------------
**In [38]**  **Second set of images**


![alt text][image3801]  
  52.34% Right-of-way at the next intersection
  34.86% Dangerous curve to the right
   4.78% Beware of ice/snow
   4.51% Priority road
   1.76% End of no passing by vehicles over 3.5 metric tons



![alt text][image3802]  
  99.63% Speed limit (30km/h)
   0.35% Speed limit (20km/h)
   0.00% No passing for vehicles over 3.5 metric tons
   0.00% Right-of-way at the next intersection
   0.00% Roundabout mandatory



![alt text][image3803]  
  58.45% Keep right
  34.81% Yield
   2.19% Go straight or right
   1.97% End of all speed and passing limits
   1.42% Turn left ahead



![alt text][image3805]  
  96.62% End of speed limit (80km/h)
   2.61% End of no passing by vehicles over 3.5 metric tons
   0.74% Roundabout mandatory
   0.01% Priority road
   0.00% Right-of-way at the next intersection



![alt text][image3806]  
  99.97% Keep right
   0.01% Turn left ahead
   0.00% Go straight or right
   0.00% Roundabout mandatory
   0.00% End of all speed and passing limits



![alt text][image3807]  
  97.23% Keep right
   2.60% Traffic signals
   0.10% End of all speed and passing limits
   0.04% Go straight or right
   0.00% Turn left ahead


# Second set of Images
Interesting results on this dataset,  

(**TOODO Finish**)

Remember none of these images were in the training set; the correct answer is not in the list of labels provided.  
So it was impossible for our classifier to get these correct (except, sort of, on the first image..)  
- It's interesting to me that on the 1st image, it did manage to locate the "sub-sign" within it, as 5th prob'  
- The 2nd image also chose what I consider the closest two answers as it's top two  
-- though the difference in confidence is vastly different, and perhaps swapped from what might be expected.  
- The 3rd, 5th,6th images do NOT focus on the Number depicted, which is what *I* do when interpolating their meaning.
--  It surprises me that it does not choose 
- The 4th image, however, does seem to consider the number, and the "not / end-of" in it's top two choices
"""
