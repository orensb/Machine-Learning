# Image Classification

 I built a convolutional neural network that can predict whether two shoes are from the same pair or from two different pairs.

 ## The Data
 - There are three main folders: train, test_w and test_m.
 - The dataset is comprised of triplets of pairs, where each such triplet of image pairs was taken in a similar setting (by the same person).
 - Every image is a numpy array of shape [224, 224, 4]

## Data Preparation 
- Reshape the data in the shape we want:  [N, 3, 2, 224, 224, 3]
  * N - the number of triplets allocated to train, valid, or test
  * 3 - the 3 pairs of shoe images in that triplet
  * 2 - the left/right shoes
  * 224 - the height of each image
  * 224 - the width of each image
  * 3 - the colour channels
- Labelled training data, with same and different pair (positive and negative examples).
- Creating the correct numpy by concatenated the right and left shoe , and for negative 2 difeerent one

## CNN Model
- CNN model in PyTorch called CNN that will take images of size  3×448×224 , and classify whether the images contain shoes from the same pair or from different pairs.
- We can find the model in the Function.py file:
  * Conve2d
  * padding and stride
  * Linear
  * Dropout
  * BatchNorm2d
  * relu
  * max_pool2d
## Channeled CCN
-  I will first manipulate the image so that the left and right shoes images are concatenated along the channel dimension.
-  The input to the first convolutional layer will have 6 channels instead of 3 (input shape  6×224×224 ).

## Traning
- Criteria : Cross entorpy loss
- Optimizer : Adam
- Hytper paramter : Learning rate , step size , gamma , weight decay -> has been chosen by simulation result
- Batch size of 32 and epochs of 30
- For each epoch and batch size we train the model, with backward and step function, and our goal is to minimize the loss function
- Validate our traning each epoch to estimate the correctness of the traning curve 
