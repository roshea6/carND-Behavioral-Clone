from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda
from keras.layers import MaxPooling2D, Cropping2D, ELU, Dropout
import numpy as np
import cv2
import csv

# Lines from the csv file
lines = []

with open('./data/driving_log.csv') as csv_file:
    # CSV file reader object
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)
        
# Empty lists for our training images and steering angles
images = []
angles = []

# Correction factor using left and right images
correction = .20

# Loop through all of the lines from the csv file
for i, line in enumerate(lines):
    # Skip the first iteration 
    if i == 0:
        continue
        
    # Get the center camera image
    source_path = line[0]
    # Split the path into a list of individual directories and the file name at the very end
    filename = source_path.split('/')[-1]
    img_path = './data/IMG/' + filename
    img = cv2.imread(img_path)
    images.append(img)
    
    # Get the stearing angle from the line and store it as a float
    angles.append(float(line[3]))
    
    # Get the left camera image
    source_path = line[1]
    # Split the path into a list of individual directories and the file name at the very end
    filename = source_path.split('/')[-1]
    img_path = './data/IMG/' + filename
    img = cv2.imread(img_path)
    images.append(img)
    
    # Get the stearing angle from the line and store it as a float and add the correction factor
    # This simulates the car needing to turn right
    angles.append(float(line[3]) + correction)
    
    # Get the right camera image
    source_path = line[2]
    # Split the path into a list of individual directories and the file name at the very end
    filename = source_path.split('/')[-1]
    img_path = './data/IMG/' + filename
    img = cv2.imread(img_path)
    images.append(img)
    
    # Get the stearing angle from the line and store it as a float and subtract the correction factor
    # This simulates the car needing to turn left
    angles.append(float(line[3]) - correction)
    
    
# Flip all the images and labels in the training set so the dataset has equal left and right turning examples
aug_imgs = []
aug_angles = []
for img, angle in zip(images, angles):
    # Append the normal image and angle
    aug_imgs.append(img)
    aug_angles.append(angle)
    
    # Append the flipped images and angles
    aug_imgs.append(cv2.flip(img, 1))
    aug_angles.append(angle * -1.0)
    
    
# Convert the images and angles to numpy arrays so they can be used to train the network
x_train = np.array(images)
y_train = np.array(angles)

# Create the Nvidia driverless car model
model = Sequential()

# Input lambda layer to normalize and mean center the images
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))

# Cropping layer to take only the ROI (Removes top 70 and bottom 25 pixels
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Convolution layer 1 with 24 5x5 filters
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))

# Convolution layer 2 with 36 5x5 filters
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))

# Convolution layer 3 with 48 5x5 filters
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

# Convolution layer 4 with 64 3x3 filters
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Convolution layer 5 with 64 3x3 filters
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Flatten the output of the last convolution layer so it can be connected with a fully connected layer
model.add(Flatten())

# Fully Connected layer 1
model.add(Dense(100))
# model.add(ELU())
model.add(Dropout(.5))

# Fully Connected layer 2
model.add(Dense(50))
# model.add(ELU())
model.add(Dropout(.5))

# Fully Connected layer 2
model.add(Dense(10))
# model.add(ELU())

# Output single node (The steering angle)
model.add(Dense(1))

# Compile the model with the proper loss function and optimizer
model.compile(loss='mse', optimizer='adam')

# Train the network
model.fit(x_train, y_train, validation_split=.2, shuffle=True, epochs=7)

model.save('newest_model.h5')
          
      




