# **Behavioral Cloning** 
Ryan O'Shea

IMPORTANT: The videos I captured of the car driving around the track were too large to be included so they were uploaded to Youtube and linked below.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./ex_imgs/nvidia-architecture.png "Model Visualization"
[bad_corner]: ./ex_imgs/bad_corner.png "Grayscaling"
[left]: ./ex_imgs/left.jpg "left camera"
[center]: ./ex_imgs/center.jpg "center camera"
[right]: ./ex_imgs/right.jpg "right camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* best_model.h5 containing a trained convolution neural network 
* project_writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py dropout_model.h5
```

A new model can be generated using the following command
```sh
python model.py
```

This will save a new model file under newest_model.h5. The grader is not likely to have the dataset I generated so it's recommended to use the premade model file.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Almost every step of the code is commented to increase readability for other users as well as to rienforce my own understanding of the various concepts covered by this project. Comments on the individual network layers especially helped me to increase my understanding of neural network structures.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 120-159). This model is based on the Nvidia Convolution neural network in their End to End Learning for Self-Driving Cars paper which can be found here http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf. A visualization of this model can be seen in the image below. The normalization layer that the NVidia model uses was implemented as a Keras lambda layer that normalizes and 0 centers the image. A cropping layer 

![alt text][model]

The model uses ReLU activation on each of the Convolutional layers. A different activation type, exponential linear unit (ELU), was tested after I noticed it while going through the Keras documentation but it did not increase performance of the network so ReLU was kept as its more popular and I understood it better. The second layer is a cropping layer that removes the top 70 and bottom 25 pixels of the image. This was to used to exluded to superfluous image data that included not road terrain and the hood of the car which could cause the networking to learn bad features from the complex background. The first two laters can be seen in the code block below.

```python
# Input lambda layer to normalize and 0 centers the images
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))

# Cropping layer to take only the ROI (Removes top 70 and bottom 25 pixels
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
```

#### 2. Attempts to reduce overfitting in the model

Dropout layers were added throughout the network with varying levels of effectiveness. They were primarily used in the fully connected layers with varying keep rates that were tested and tweaked over time.  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The Keras Sequential object's fit function was used to train the model. The validation_split and shuffle variables were used to create a randomly chosen validation set to be completely different from the training date. In this case the validation set used the standard 20% of the training data and saw good results. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. There are some point where the car gets close to leaving the track or swerves a bit but it always returns to where it should be without leaving the track hitting any obstacles.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 165). The number of training epochs, keep rate, and side camera correction factor were the primary parameters that needed to be tuned. Epoch number and keep rate were fairly quick to tune but the camera correction factor took a while to finally find a good values. I tested values between .15 and .35 and saw fairly large changes in performance of the model. When the values were too high the car would occasionally turn right off the road in the beginning. .2 Proved to be a good value but this can almost certainly be tuned more. 

#### 4. Appropriate training data

The generation of training data generally followed the guidelines provides during the lectures. 3 laps of smooth center lane driving were recorded as the bulk of the training data then a supplementary lap of recovery behaviors from the left and right side of the lane were recorded. To record the recovery the car would be position near the edge of the road before recording was started and then driven back to the center. The recovery lap had extra attnetion given to the curves of the trakc because they seemed to be where the car was struggling the most. Unfortunately this data proved insufficient as the car always failed at the same place. The curve after the bridge with the dirt road was one of the hardest areas for the model to overcome. This area can be seen in the image below. The car would often begin its turn and then stop turning and go straight onto the dirt road. To overcome this addition training data of just that curve was recorded. I made sure to perform incredibly smooth and generous turn for the first few runs and then recorded recovery behaviors specific to that turn. This included almost going onto the turn road then making a hard left turn to stay on the proper path. 

![alt text][bad_corner]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The choice of model, edits to it, and how it was training are explained above.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Examples of this can be seen in the following videos.

- Original working model: https://www.youtube.com/watch?v=u9dNdSKmqv0&t=217s 
- Improved Model with dropout layers: https://www.youtube.com/watch?v=bmcW9ThK8CQ&t=46s

The first video is of a model training without any dropout layers in the network. This performance was good but the turn in on the curve after the bridge was not very smooth and I felt like it could be much better. To do this dropout layers were added in the fully connection portion of the network to prevent overfitting. Thie improved performance noticably especially near the dificult curve which is now completed in a much nicer looking manner. This is likely due to the dropout layers helping the network to become more robust overall. 

#### 2. Final Model Architecture

The final model architecture is shown above in the previous sections. The Nvidia architecture that was shown in the lecture performed very well on this project. With the initial training data with zero efforts it performed fairly well but often ran into problems on the bridge and in the curve directly after it. The modifcation to the network and the generation of additional training data helped improve performance significantly. There are likely other networks out there that perform even better than the one used here. Deeper and more complex networks would likely be able to extract more data from the environment that can be used in the decision making of the model. However, for the purpose of this project this model still performed very well. 

The model was tested on the second track but did not perform well. It was able to perform some parts of the track but the presence of shadows and the complex nature of the track was too much for the current model. To improve this the model should be trained on both tracks in order help it generalize better as well as to prevent the model from just learning the first track. Some extra preprocessing would also likely help the model generalize more.

#### 3. Creation of the Training Set & Training Process

The recording of training data is discussed in the previous sections. Image data was captured from the left, center and right cameras of the car which can be seen below in that order.

![alt text][left]
![alt text][center]
![alt text][right]

The use of the left and right cameras in addition to the steering correction factor was incredibly helpful in generation more "turning" data for the model to train on. Without this the model would be heavily skewed towards having mostly 0 turning angle data due to the nature of the track. This also tripled the size of the initial dataset which was helpful for generating lots of data but from only a few minutes of video time.

To augment the dataset, I also flipped images and angles. This helped give the network more "right" turn data to train on to make it even more robust. This effectively doubled the size of the dataset. The code for flipping the images can be seen below.

```python
aug_imgs = []
aug_angles = []
print("Generating flipped images")
for img, angle in zip(images, angles):
    # Append the normal image and angle
    aug_imgs.append(img)
    aug_angles.append(angle)
    
    # Append the flipped images and angles
    aug_imgs.append(cv2.flip(img, 1))
    aug_angles.append(angle * -1.0)
```

There was no formal data preprocessing step as the network had built in preprocessing with its normalization layer. Some data preprocessing may have been good though and would be a valuable area to explore in the future. While the neural network extracts it's own features it would be interesting to see if some of the preprocessing techniques from the class' earlier projects would be useful here. 

## Acknowledgements
The documentation for the opencv and Keras python libraries were used extensively for this project. The lecture notes and videos were also referred to often and directly used in parts of the project. Some stack overflow questions were used to fix roadblocks I hit with general programming issues. 
