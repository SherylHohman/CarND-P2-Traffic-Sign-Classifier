# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / WRITEUP.md (Readme file for WriteUp)

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SherylHohman/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set 

**In [106]** uses _numpy_ library togather basic statistics about the dataset 

I also used a [routine](http://stackoverflow.com/a/850962/5411817) to read the number of output classes directly from the signnames.csv file.

```
Number of training examples   = 34799  
Number of validation examples = 4410  
Number of testing examples    = 12630  
Image data shape = (32, 32, 3)  
Number of classes = 43  
```

#### 2. Include an exploratory visualization of the dataset 

##### **In [3]** and **In [80]** visualizally explore the data set.  

**In [3]** displays a bar chart showing distribution of the data  
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

### Design and Test a Model Architecture

#### 1. Identify where in your code, and Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

### PreProcessing

**In [5]**, **In [6]**, **In [7]**, **In [8]**, **In [9]**, **In [73]**  
Contain various routines for pre-processing images.   Mostly they focus on various   
ways of grayscaling, and mean-centering the distribution of pixel values.  

#### **Grayscaling**  
##### **..what did NOT work**  
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

#### **Normalizing**  
##### **..what did NOT work**   
**In  [8]** `get_per_channel_mean_zero_centered_datasets()`   
**In [84]** `get_normalized_images()`   
I tried normalizing the pixel values across the entire training dataset, and I also tried normalizing per channel.  Both methods were unsuccessful.  Although I these were the most common normalization techniques I read about, when I trained on these preprocessing techniques, the results were miserable - looking about the same as an untrained network.  
Perhaps I implemented them incorrectly.  
In retrospect, I believe the use case for that technique is for comparing, say frames from a security camera.  In that case the exposure for images is the same from frame to frame (changes with time of day).  What is different is brightness/contrast etc in different parts of the image.  
In our case, the images are taken from many different exposures, lighting conditions, color casts, etc. They are taken in different physical locations. So in this way, there would not be a uniformity across all images in the training set that we should try to normalize on.  Instead, the images are zoomed in, and while they may contain shadows cutting across an image that could "confuse" the network, generally they are fairly uniform within an image. And shadows, etc are features that we want to train on anyway, as they are going to occur "in the wild".  We want our network to recognize a sign whether it has a shadow cutting across it or not.  
This is my reasoning why the per channel, and per training set normalization techniques did not work.  

#### **..and why Per-Image Normalization DID work.**   

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

##### **No images to display here.**  
In normalizing my data, "pixel values" were turned into values between -1 and 1.  
  
Hence the arrays representing my images'pixel values are no longer in the 0-255 range for displaying images.  
Perhaps there is a library function that can display such values as an image. I did not see one.  And decided it unnecessary to try viewing the resulting transformation visually. ;-)  


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)  

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

I Chose to Implement a LeNet Architecture for my Model

#### LeNet Architecture
**In [75]** is LeNet architecture  
**In [12]**, **In[13]** contain implementation details for conv and fcc layers  

```  
Layer                     In-Shape      Out-Shape   Description / Settings    Function Used  
                                                                                             
Input Layer                           (32, 32, 3)  RBG Image, Preprocessed  
                                                                                             
Layer 1:  
convolution             (32, 32, 3)   (28, 28, 6)  filter=?,?  stride=1,1     tf.nn.conv2d()  
activation   RELU       (28, 28, 6)   (28, 28, 6)                             tf.nn.relu()  
pooling      Max Pool   (28, 28, 6)   (14, 14, 6)   ksize=2,2  stride=2,2     tf.nn.max_pool  
                                                                                             
Layer 2:  
convolution             (14, 14,  6)  (10, 10, 16) filter=?,?  stride=1,1     tf.nn.conv2d()  
activation   RELU       (10, 10, 16)  (10, 10, 16)                            tf.nn.relu()  
pooling      Max Pool   (10, 10, 16)  ( 5,  5, 16)  ksize=2,2  stride=2,2     tf.nn.max_pool()  
                                                                                             
Flatten:                ( 5,  5, 16)         (400)  
                                                                                             
Layer 3:  
fully connected                (400)         (120)                            prev * weights + bias  
activation    RELU             (120)         (120)                            tf.nn.relu()  
dropout                        (120)         (120)  keep_probability=0.5      tf.nn.dropout()  
                                                                                             
Layer 4:  
fully connected                (120)          (84)                            prev * weights + bias  
activation    RELU              (84)          (84)                            tf.nn.relu()  
dropout                         (84)          (84)  keep_probability=0.5      tf.nn.dropout()  
                                                                                             
Layer 5:  
fully connected                 (84)          (43)                            prev * weights + bias  
logits  
                                                                                             
either: 
sparce_softmax..      
softmax                     

```  
sparce_softmax_.... (**TODO**)

**In []**  Evaluation, backprob=p.. **(TODO)**


 Each Layer begins with all bias elements initialized to 0 
 and weights randomly distributed along a normal distribution ??centered??(mean) at zero, and standard deviation of 0.1 ?? aka in the range of +- 0.1 ?? (**TODO**)  


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.  

(**TODO**)
The code for training the model is located in the eigth cell of the ipython notebook.  

To train the model, I used an ....  

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.  

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook. 
(**TODO**) 

My final model results were:  
* **In[26]** **In[23]** training set accuracy of ?  

* **In[26]** **In[23]** validation set accuracy of ?   

* **In [27]** test set accuracy 
```
evaluating..  
0.940063341062  
Test Loss     = 0.632  
Test Accuracy = 0.940  
```   
![alt text][img]

Training Dataset statistics, and charts are located in the ./training_stats directory  
 The first two models that I saved data on, I will not discuss here.  They are saved under the names "training_stats_170325" (png and txt), and "training_stats_170326-02_perchannel_mean" (png and txt) in the training_stats directory.  

Here I will compare three "successful" (ie achieved 93%) versions of my network.  
They all used the same 'get_per_image_mean_centered_training_sets' function for pre-processings the images.  The difference is in architecture, and training parameters values.

**Version 1** (aka Model 3) achieved 93% validation accuracy, though it oscillated above and below this number.  

**Version 2** (aka Model 4) achieved 95% validation accuracy, just by changing the learning rate, (and sigma also??). I was unhappy with climbing Validation Loss alongside the Perfect Training Loss and Accuracy. Validation accuracy was either improving or unchanging on average, though it made big drops occassionally before climbing back.  

**Version 3** (aka Model 5) added dropout layers, and also achieved 95% validation accuracy. this time the validation accuracy remained clearly above the 93% minimu, was steadier overall, had much better Validation Loss, and was not overfitting terribly.

##### version 3, Model 3 (descent, bordeline passing 93%)  
![alt text][image263]   
From this chart, I decided to adjust the learning rate (?and sigma) ..

#### version 4, Model 4 (good, passes kinda 95%)  
![alt text][image264]  
Clearly, this model responded well to the new learning rate (and sigma)
Seing.. I'm curious if It might be helpful to adjust learning rate after Epoch..  
Or to increase learning rate for the initial ??n Epochs, in order to reduce the amount of time spent training.
More importantly, however is The steady climb in validation, after it's initial drop.  
While the training loss continues to decrease (to Zero! - a sure sign that it has overfitted the training data),  
the loss in the validation set climbs with every Epoch.  Even while validation accuracy continues to get better.
Indicated to me that my model was over fitting to the test set.
To address this I considered adding a dropout to ny neural net, and adding data augmentation.  
I decided to first try dropout, as it was more straigntforward - only 1 paramater to tune.
Data Augmentation can take many forms. Which techniques to use, how many techniques to apply at a time, etc. It would also take more work to implement each of the augmentation alogrithms.  
And it's best to implement a single technique at a time, to see how the network responds to each.  

I decided add dropout.

#### version 3, Model 5 (dropout: clearly 95%)   
![alt text][image265]  
I added dropout to the first two fully connected layers: layer3 and layer4.  
layers 3 and 4 now consisted of fcc_layer, relu_activation, and finally, dropout.
The keep probability I used for each was 0.5.  This appeared to be a standard starting value to use, and it seemed to work well for me.  Other options could have been to use dropout on just one of the two layers, or to vary one or both keep_probabilities, perhaps trying 0.25 for one of them.  I was happy with improved results obtained with 0.5 keep at layer3 and 0.5 keep at layer 4.  
The network took longer to train, and in fact, from the training_stats chart, it does not appear to have reached it's maximum potential at 100 Epochs.  However, it is fairly flat; ROI in additional Epochs would be low.
Again, it might be helpful to lower learning rate after Epoch..  
It also might be helpful to increase learning rate for the first ?nn Epochs.
As this model takes longer to train, this is the first adjustment I would attempt.  
After that I would tackle Augmentation.  

Looking at the chart of the training_stats, We immediately see the effects of adding dropout.
In this new chart, we see that once achieved, the validation accuracy remains above ??%.  
In contrast, the previous model would routinely drop to about ??93%
While the magnitude of oscillation is shallower (this is good), the frequency is greater.  This is to be expected, as we are throwing away half of the information it trained on at each step.  
Result: it takes longer to train, but the overall gains are more steady or stable.  Basically, like an electric circuit, any particular electron may move forward or backward, and only a tiny amount at that, overall there is a steady current.  Likewise, there is a steady climb on average.
We also see that Validation Loss has decreased overall! And while it still climbs over time, the rate of climb is Much Slower.  Finally, we see that Validation loss is much closer in distance to Training loss, and its slope is closer to Training loss as well.  Training loss does Not reach Zero, Training Accuracy does Not reach 100%.  
I conclude that dropout was effective in reducing the overfitting problem seen in the previous model.  
While tuning the dropout_keep rate might gain further improvements, that does not seem to be the best use of effort or time at this point: overfitting is under control with respect to overall accuracy of our model.  
At this point, Accuracy, or ability of our model to learn particulars of each sign appears to be a reflection of our training data itself.  
Refering back to our distribution of data, we can see that there is a vast variation in the number of examples provided for each label.  For example, label ??n has around ??n example images, while ??m has around ??n images.  
Additionally, looking at just 3 of our training images, we see that the quality of images in that sample set varies widely: one is severly under exposed, another is overexposed - I do not even know what traffic sign that one represents.  It seems to me data augmentation would be a great next step to improve our network.  
In the next section, when I throw new images at it, this becomes even more apparent.  I chose images that were taken at an angle, images with bad cropping, and images that were under represented in the training data.  The network performed rather poorly with these.  My images likely deviated greatly from the type of labeled data it had seen so far.  
Indeed, the test set achieved 94% accuracy.  My images achieved only ??n% accuracy.  Of course, my sample set was too small to report a reliable accuracy report.


* What are some of the important design choices and why were they chosen?  For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  

If a well known architecture was chosen:  
* What architecture was chosen?  
* Why did you believe it would be relevant to the traffic sign application?  
(**TODO**)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
 

### Test a Model on New Images  

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.  

Here are images that I found on the web, grouped into 2 Sets:

##### **The first set**:  
![alt text][image301]  
These are relatively straight forward - all these signs exist in the training set and map our classifier.  
However, several are cropped poorly, are taken at an angle (as opposed to straight on), and one lacks color information.  Some of these traffic signs had many examples in the training set, some had few examples to train on.  How well does the network  perform on these images ?  


##### The second set:  
![alt text][image302]    
This set consists entirely of traffic signs that do not exist in our classification data.  
In other words, it is impossible for our network to obtain the correct answers for these signs.  
I chose them to get an insight on how it sees and correlates traffic sign images, vs how I see and correlate traffic sign images.  As I am not familiar with German Traffic signs, I also did not know the true meaning of these signs.  Never-the-less, even though none of these signs were in our classifier, if I had to choose one or more of the class ids as a guess, would it match the guess made by the network ??  
Did it learn to "see" the same "features", and map these to labels in the same way that I did ?  


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).  


#### Here are the results of the prediction:  
**In [33]**  

![alt text][image301]  
```  
 n Actual Sign              Predicted Sign                              Correct?
   -----------              --------------                              --------
 1 Stop Sign                Road work                                     no
 2 Stop Sign                Speed limit (50km/h)                          no            
 3 No Entry                 End of all speed and passing limits           no    
 4 General Caution          Speed limit (50km/h)                          no            
 5 General Caution          General caution                               YES        
 6 General Caution          General caution                               YES        
 7 Left Turn Ahead          Turn left ahead                               YES        
 8 No Overtaking by Lorries No passing for vehicles over 3.5 metric tons  YES            
 9 Speed Limit 100km/h      Speed limit (80km/h)                          no           
10 Speed Limit 100km/h      Speed limit (60km/h)                          no     
```        

The model incorrectly guessed 6/10  and correctly guessed 4/10.  
That is an Accuracy of only 40 %  

If choose to look at only 1 of the predictions on signs where I provided multiple crops,  
I get a top accuracy of : 50%, at 3 wrong, 3 correct = 3/6 = 50%
A low accuracy of ..... : 33%, at 4 wrong, 2 correct = 2/6 = 33%

This is Much lower than the accuracy from the test set.  
Conclusion: The training and test sets are not representative of the images I provided.  The model may have preferrred signs that had higher representation in the training set.  Statistics may not be as accurate for such a small sample size.  
 
**Set_1**: My classifier gave mixed results.  It performed terrible on several images I expected good results on, though it also got a couple correct.  It performed best on images with a straight on camera angle with a, tight, crop.  

Several of the images I provided were poorly cropped, taken from an angle rather than straight on, and one was even lacking color information, which is something that I trained on.  The (second) speed limit 100km/h was a good crop, so I'm rather surprised at it's mistake on this one.  

Signs 1, 2: I expect should get a correct answer with closer cropping.  
    I was overly optimistic that it would do well even given the bad crop I handed it.  
    I can imagine that the crop I gave it lent a partial figure-8 shape to confuse it, though at the wrong scale??  
    
Sign 3: Was in black an white, so it is missing color information, which may have otherwise helped nudge it toward the correct answer.  
- Additionally, the sign was skewed at an angle, and placed against another sign,  
- so it's overall shape could be difficult to discern.  
- Unfortunately, the misinterpretation is Grave:   
  - 'No Entry' sign became an "End All Speed and Passing Limits"  
  - **Nearly the Direct Opposite in Intention, Meaning, desired driver Behaviour  !**  Very Bad Result, indeed.    
        
Sign 4:  No Idea what happened there! Perhaps the edges of the triangle resembled a 5, though at the wrong scale??  
    If so this is similar to what I "imagine" threw it off for image 6. Or I just have an imagination.      
Sign 5 and 6: A closer crop, Removing just a portion of the sign underneath made all the difference.  

Signs 7, 8: Yea! it got those correct.  

Signs 9, 10: It was correct in that it's a speed limit sign.  Surprised that it mistook 100 for 60 or 80.  Perhaps the poor crop in #9 lended a portion of an "8" shape?  However, that in no way explains why the close crop generated a "60" interpretation.  Perhaps 100km was poorly represented in the training set.  

So this classifier was correct on 40% of images (33% or 50% if look a single image from each class).  
That's Far lower than the test, and validation sets!!  It's downright Terrible.   



**Set_2**: consisted of street signs that were not part of the street sign names my classifier was trained on.  It was impossible for my classifier to get any of these correct.

- The correct sign names do not even exist in the csv signnames file,   
- I chose these out of curiousity about how it trained on features, as compared to how *I* see features.  
- would see the same type similarities **I** see when looking at them?  
- Would it make the same guesses I did (not knowing German signs) ?  


![alt text][image302]  
```
Right-of-way at the next intersection  
Speed limit (30km/h)  
Keep right  
End of speed limit (80km/h)  
Keep right  
Keep right  
```    
Sign 2: Surprisingly, it got almost about as close a guess as possible.   Must be beginner's luck.  

Sign 5: Was also surprisingly good guess. 

Signs 3, 6, 7: These three signs I Expected the classifier to, no-brainer, return specific predictions.  
    Nope. It passed over images which I think they look Very similar to, 
    in favor of images that I think look not at all similar to these signs.   What was its "thinking" process ??  dI immediately see numbers, and would associate these signs with speed limit signs, or end of speed limit signs.  Something that had a number front and center.  
    
---------------------------------------------------



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)  

The code for making probability predictions on my final model is located in the 11th cell of the Ipython notebook.  
(**TODO**)


**In [39]**  **First set of images**
(**TODO: finish**)

![alt text][image3901]  Stop Sign  
```  
  99.17  Road work  
   0.34  Speed limit (30km/h)  
   0.26  Stop  
   0.20  Yield  
   0.00  No entry  
```  
Stop sign was this networks 3rd choice, but at less than 0.3% confidence.  
It was 99.2% confident this was a Road Work sign.  
"Stop" is a Very important sign to be correctly identified !!
The sign I supplied was taken at an angle, which would make the sign more difficult to recognize if it was trained only on images taken head on.  
Because the image was taken at an angle, the crop also was necissarily bad.  This means that "background" information was not uniform around the image, and would more likely be interpreted as features it should try to interpret.
**TODO** see how well stop was represented in training data  


![alt text][image3902]  Stop Sign  
```  
  54.89  Speed limit (50km/h)  
  21.63  Speed limit (70km/h)  
  10.41  Speed limit (30km/h)  
   9.42  Speed limit (80km/h)  
   3.58  Stop  
```  
Stop was this networks 5th choice, but this tine it's confidence increased 10-fold to 3.6%  
It is the same photo taken at an angle, but more closely cropped.  In fact, this cropping cut out some of the sign edges.  It still has much background information on the left of the image.  
The poor cropping made the less confident of any particular selection overall, thereby increasing it's confidence in it's guess for a Stop sign. While it's highest confidence in _any_ sign dropped to 55%, it is about 96.4% certain that it's a speed limit sign of some sort.  It is interesting to see how much of a difference a small change in a bad crop made on the networks top predictions and uncertaintly levels. I wonder how will it would have performed on a crop that removed all background info, leaving it primarily with a skewed image of the words "Stop" would influence it's predictions.  Even better, I wonder if augmenting the training set with skewed images, rotated images, or traslated images (shifting the image over, cutting part of it off, leaving a line of no data on one side) would affect its performance on this image, and its overall performance for all images.

![alt text][image3903]  No Entry  
```  
  83.94  End of all speed and passing limits  
   7.62  End of no passing  
   4.50  Ahead only  
   1.26  Go straight or right  
   0.93  End of speed limit (80km/h)  
```  


![alt text][image3904]  General Caution  
```  
  62.78  Speed limit (50km/h)  
  36.94  Speed limit (30km/h)  
   0.18  Wild animals crossing  
   0.08  Double curve  
   0.00  Dangerous curve to the left  
```  


![alt text][image3905]  General Caution    
```  
  99.99  General caution  
   0.00  Traffic signals  
   0.00  Road work  
   0.00  Bumpy road  
   0.00  Road narrows on the right  
```  


![alt text][image3906]  General Caution    
```  
  99.93  General caution  
   0.06  Traffic signals  
   0.00  Road narrows on the right  
   0.00  Road work  
   0.00  Beware of ice/snow  
```  


![alt text][image3907]  Left Turn Ahead  
```  
 100.00  Turn left ahead 
   0.00  Yield  
   0.00  Go straight or right  
   0.00  End of all speed and passing limits  
   0.00  Ahead only  
```  


![alt text][image3908]  End of No Passing by Lorries (vehicles over 3.5 metric tons)
```  
 100.00  No passing for vehicles over 3.5 metric tons  
   0.00  Priority road  
   0.00  End of no passing by vehicles over 3.5 metric tons  
  -0.00  Speed limit (20km/h)  
  -0.00  Speed limit (30km/h)  
```  


![alt text][image3909]  Speed Limit 100km/h  
```  
  76.27  Speed limit (80km/h)  
  11.34  Speed limit (60km/h)  
  10.05  Speed limit (50km/h)  
   1.51  Stop  
   0.73  No vehicles  
```  


![alt text][image3910]  Speed Limit 100km/h     
```  
  99.52  Speed limit (60km/h)  
   0.47  Speed limit (80km/h)  
   0.00  No passing for vehicles over 3.5 metric tons  
   0.00  No passing  
   0.00  Turn left ahead  
```  

---   

(**TODO: finish**)
- Interesting how cropping changed the predictions.   
 -- Good Cropping matters. (Giving my trained model, anyhow)
 -- Stop Sign: better crop (tho not great), stop was rated 3rd on the list, but confidence was lower. worse crop, was > 10 x more confident in "stop" as a choice, though it was now 5th. Of course, this is also reflected in that the worse crop lowered it's confidence in anything overall. Better crop gave it a 99% certainty in a wrong answer, vs a 55% top certainty in the close cropped version. 
- I'm surprised that the 100km/h speed limit did not even make the list, even with good cropping.  
- How did the second to last image have 2 predictions at less than 0 % ??  
 --Is this a red-flag that something is wrong (with alogrithm) ??  Or that rounding hit an overflow ??  


**TODO** probably won't, but these would be interesting charts to add:
Training data: signs with number of example images:
above 1250
 750-1250
 300ish-750
 under 250 or 300
 
 count signs:
 create list of signs falling in said count range
 print the lists
 
  read all signs in top 5 predictions
  display their representation in the dataset
     (it total number of images, or % of dataset that was that sign,
      or where it is in the standard deviation of images
      or which count range it was in
      
  using the 6 signs I gave it,
      display their representation in the dataset
      ( total number of images, or % of dataset that was that sign,
      or std
      or which count range it was in

Perhaps add display a chart comparing those stats  


--------------------------------------------------------------------------------
#### **In [38]**  **Second set of images**

![alt text][image3801]    
```  
  52.34  Right-of-way at the next intersection  
  34.86  Dangerous curve to the right  
   4.78  Beware of ice/snow  
   4.51  Priority road  
   1.76  End of no passing by vehicles over 3.5 metric tons  
```  
  
![alt text][image3802]    
```  
  99.63  Speed limit (30km/h)  
   0.35  Speed limit (20km/h)  
   0.00  No passing for vehicles over 3.5 metric tons  
   0.00  Right-of-way at the next intersection  
   0.00  Roundabout mandatory  
```  
  
![alt text][image3803]    
```  
  58.45  Keep right  
  34.81  Yield  
   2.19  Go straight or right  
   1.97  End of all speed and passing limits  
   1.42  Turn left ahead  
```  
  
![alt text][image3805]    
```  
  96.62  End of speed limit (80km/h)  
   2.61  End of no passing by vehicles over 3.5 metric tons  
   0.74  Roundabout mandatory  
   0.01  Priority road  
   0.00  Right-of-way at the next intersection  
```  
  
![alt text][image3806]    
```  
  99.97  Keep right  
   0.01  Turn left ahead  
   0.00  Go straight or right  
   0.00  Roundabout mandatory  
   0.00  End of all speed and passing limits  
```  

![alt text][image3807]    
```  
  97.23  Keep right  
   2.60  Traffic signals  
   0.10  End of all speed and passing limits  
   0.04  Go straight or right  
   0.00  Turn left ahead  
```  

Interesting results on this dataset,  
(**TODO Finish**)

None of these images were in the training set; the correct answer is not in the list of labels provided.  
So it was impossible for the classifier to get any of these correct (the first image is a minor exception)  
- It's interesting to me that on the 1st image, it did manage to locate the "sub-sign" within it, as 5th prob'  
- The 2nd image also chose what I consider the closest two answers as it's top two  
-- though the difference in confidence is vastly different, and perhaps swapped from what might be expected.  
- The 3rd, 5th,6th images do NOT focus on the Number depicted, which is what *I* do when interpolating their meaning.
--  It surprises me that it does not choose 
- The 4th image, however, does seem to consider the number, and the "not / end-of" in it's top two choices
"""  

### (Optional) Visualize the Neural Network's State with Test Images  
Hmm.. Unfortunately, I don't think I can use the function above to gain insight on training features. Not only are my tensorflow training variables are encapsulated inside a LeNet(x) function. So the tensor I need to pass into the outputFeatureMap function are not global variables. I have no access or handle to them from here, or anywhere outside that function.

If it is indeed possible to access the required variable, I would be interested in gaining insight, for about 4 images. That is not going to happpen at this time, however. 



------------------------------------------------------------------------------

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Traffic Sign 4"  

[image3]: ./data_plotted_image_distribution_amongst_classes.png "Visualization_1: Distribution of Images between classes, and across Training, Validation, and Test Sets"  
[image80]: ./writeup_sample_images_cropped/sample_traffic_signs_from_training_set.png "Visualization_2: Sample Images from the Training Set"  
[image5]: ./writeup_sample_images_cropped/sample_grayscale-1channel_conversion.png "sample image converted to 1D grayscale"  
[image6]: ./writeup_sample_images_cropped/sample_grayscale_conversions_single_channel.png "sample images converted to 1D grayscale"  
[image7]: ./writeup_sample_images_cropped/sample_grayscale_1D_to_3D_conversion.png "grayscale image converted from 1D to 3D loose tonal benefits gained in the rgb to gray conversion"  
[image12]: ./writeup_sample_images_cropped/sample_traffic_signs_from_training_set.png "Visualization 2"  

cell 26: cell Training Stats

[image265]: ./training_stats/training_stats_170406_2032.png "Training Stats Model 5: with dropout"  
[image264]: ./training_stats/training_stats_170327_1518.png "Training Stats Model 4:"   
[image263]: ./training_stats/training_stats_170325.png "Training Stats Model 3:"  
[image262]: ./training_stats/training_stats_170327_0305.png "Training Stats Model 3a:"    
[image261]: ./training_stats/training_stats_170326-02_perchannel_mean.png "Training Stats Model 2:"    

cell 30 Sample Traffic Signs from the Web: Set 1, and Set 2

[image301]: ./writeup_sample_images_cropped/sample_signs_from_web_Set_1.png "Set 1: Sample Traffic Signs from the Web"
[image302]: ./writeup_sample_images_cropped/sample_signs_from_web_Set_2.png "Set 2: Sample Traffic Signs from the Web"

Set 1, cell 39 `traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames`  

[image3901]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_14-3.5-stop-1.jpg 
"14: Stop, Set1_image_1"  

[image3902]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_14-3.5-stop-2.jpg 
"14: Stop, Set1_image_2"  

[image3903]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_17-no-entry-1-bw.jpg 
"17: No Entry, Set1_image_3"  

[image3904]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_18-general-caution-1.jpg 
"18: General Caution, Set1_image_4"

[image3905]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_18-general-caution-2.jpg 
"18: General Caution, Set1_image_5"

[image3906]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_18-general-caution-3.jpg 
"18: General Caution, Set1_image_6"  

[image3907]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_34-left-turn-ahead-1.jpg 
"34: Left Turn Ahead, Set1_image_7"  

[image3908]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_42_no_overtaking-by-lorries-1.jpg 
"42: No Overtaking Lorries, Set1_image_8"  

[image3909]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_7-speed_limit_100km-1.jpg 
"7: Speed Limit 100km, Set1_image_9"  

[image3910]: ./traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/_7-speed_limit_100km-2.jpg 
"7: Speed Limit 100km, Set1_image_10"  

Set 1, cell 39   
`traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames`  
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

Set 2, cell 38:   
`traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames`  

[image3801]: ./traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/B_42_7-100kmspeedLimit_AND_no_overtaking-by-lorries-2.jpg
"Both 42, 7: 100km Speed Limit, and No Overtaking by Lorries, Set2_image_1"  

[image3802]: ./traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/N_0_2-speed_limit_25kmh-1.jpg
"N/A 0 or 2, Speed Limit 25km, Set2_image_2"  

[image3803]: ./traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/N_1-30km_Minimum.jpg
"N/A 1, Minimum Speed 30km, Set2_image_3"  

[image3805]: ./traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/N_3_6_32-end-of-speed-limit-60km-1.jpg
"N/A 3 or 6 or 32, End of 60km Speed Limit, Set2_image_5"

[image3806]: ./traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/N_5-recommended-speed-80-obsolete-1.jpg
"N/A 5, Recommended Speed 80 (obsolete), Set2_image_6"  

[image3807]: ./traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/N_6_5_32-end-of-80km-speed-obsolete-1.jpg
"N/A 6 or 5 or 32, End of 80km Speed Limit, Set2_image_7"  

Set 2, cell 38: `traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames`  
B_42_7-100kmspeedLimit_AND_no_overtaking-by-lorries-2.jpg  
N_0_2-speed_limit_25kmh-1.jpg  
N_1-30km_Minimum.jpg  
N_3_6_32-end-of-speed-limit-60km-1.jpg  
N_5-recommended-speed-80-obsolete-1.jpg  
N_6_5_32-end-of-80km-speed-obsolete-1.jpg  
