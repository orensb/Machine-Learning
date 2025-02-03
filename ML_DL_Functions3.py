


import numpy as np
import torch
import torch.nn as nn


def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 000000000


def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        #Parameters
        self.n = 5
        self.kernel_size = 3
        self.padding = (self.kernel_size-1)//2
        self.stride = 2

        # Define convolutional layers with parameters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2*self.n, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv3 = nn.Conv2d(in_channels=2*self.n, out_channels=4*self.n, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv4 = nn.Conv2d(in_channels=4*self.n, out_channels=8*self.n, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        # Define fully connected layers
        self.fc1 = nn.Linear(8*self.n*28*14, 100)  # Assuming two max-pooling layers reducing input size by half each time
        self.fc2 = nn.Linear(100, 2)  # Output layer with 2 units for binary classification
        self.drop = nn.Dropout(0.4)

        # Define Batch Norms
        self.batch_norm1 = nn.BatchNorm2d(self.n)
        self.batch_norm2 = nn.BatchNorm2d(2 * self.n)
        self.batch_norm3 = nn.BatchNorm2d(4 * self.n)
        self.batch_norm4 = nn.BatchNorm2d(8 * self.n)
        self.batch_norm5 = nn.BatchNorm1d(100)

    def forward(self, inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width

          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        out = nn.functional.relu(self.batch_norm1(self.conv1(inp)))
        out = nn.functional.max_pool2d(out, kernel_size=2)  # Max-pooling with 2x2 kernel and stride=2
        out = nn.functional.relu(self.batch_norm2(self.conv2(out)))
        out = nn.functional.relu(self.batch_norm3(self.conv3(out)))
        out = nn.functional.relu(self.batch_norm4(self.conv4(out)))
        out = out.reshape(-1,32*self.n*7*14)  # Flatten the feature maps for the fully connected layers
        out = self.fc1(self.drop(out))
        out = torch.nn.functional.relu(self.batch_norm5(out))
        out = self.fc2(self.drop(out))
        return out


        
class CNNChannel(nn.Module):
    def __init__(self):
        super(CNNChannel, self).__init__()
        # Parameters
        self.n = 5
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2
        self.stride = 2

        # Define convolutional layers with parameters
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2*self.n, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv3 = nn.Conv2d(in_channels=2*self.n, out_channels=4*self.n, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.conv4 = nn.Conv2d(in_channels=4*self.n, out_channels=8*self.n, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        # Define fully connected layers
        self.fc1 = nn.Linear(8*self.n*14*14, 100)
        self.fc2 = nn.Linear(100, 2)

        self.drop = nn.Dropout(0.3)


        # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        inp = inp.reshape(inp.size(0), 2*inp.size(1),inp.size(2)//2,inp.size(3))
        out = nn.functional.relu(self.conv1(inp))
        out = nn.functional.max_pool2d(out, self.padding)  # Max-pooling with 2x2 kernel and stride=2
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.relu(self.conv3(out))
        out = nn.functional.max_pool2d(out, self.padding)  # Max-pooling with 2x2 kernel and stride=2
        out = nn.functional.relu(self.conv4(out))
        out = out.reshape(-1,8*self.n*14*14)  # Flatten the feature maps for the fully connected layers
        out = self.fc1(self.drop(out))
        out = nn.functional.relu(out)
        out = self.fc2(self.drop(out))
        return out