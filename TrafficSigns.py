'''

http://btsd.ethz.ch/shareddata/

There are 62 labels

'''

import os

import numpy as np
import matplotlib.pyplot as plt 
#import skimage
from skimage import data

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels


DATA_PATH = "/Users/Javi/Desktop/data/TrafficSigns"
train_data_directory = os.path.join(DATA_PATH, "Training")
test_data_directory = os.path.join(DATA_PATH, "Testing")

images, labels = load_data(train_data_directory)



########################

images = np.array(images)

# Print the `images` dimensions
print(images.ndim)

# Print the number of `images`'s elements
print(images.size)

# Print the first instance of `images`
print(images[0])



######################## PLOT
'''
# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)

# Show the plot
plt.show()
'''

######################## PLOT IMAGES

# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # Pick the first image for each label
    image = images[labels.index(label)] # The method index() returns the lowest index in list that obj appears.
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
plt.show()