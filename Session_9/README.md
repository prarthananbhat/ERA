## 🤖 Assignment from Session 9 - Advanced Convolutions, Data Augmentation and Visualization

### Objectiv 🏆 
1. Make this network for CIFA10 by using Convolutions with stride 2 instead of Maxpooling
2. Reach a Receptive feild of 44 or more
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
   1. horizontal flip
   2. shiftScaleRotate
   3. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

### Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Here are the classes in the dataset, as well as 10 random images from each:

![sample_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/sample_images_downloaded.png)

### Code Base
```
.
|-- README.md
|-- S8 - Batch Normalization .ipynb
|-- S8 - Group Normalization .ipynb
|-- S8 - Layer Normalization .ipynb
|-- models.py
|-- utils.py
|-- misc
```

#### models.py
This file holds all the model definitions (network architecture). 4 models were added as part of this assignm,ent. 
1. cifar_model - *Inital model for CIFAR 10*
2. cifar_model_bn - *Model with Batch Normalization*
3. cifar_model_ln - *Model with Layer Normalization*
4. cifar_model_gn - *Model with Group Normalization*
The architecture for all the models are similar, follows C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10.

#### utils.py
This files stores the utility function. 2 additional plotting functions are added
**plot_samples_cifar** function plots 12 images from the data loader
**plot_misclassified_images** plots the 25 images that are wrongly specified with titles like *Predicted : Cat, Actual Dog*

#### Notebooks S8 - Layer Normalization .ipynb, S8 - Batch Normalization .ipynb, S8 - Group Normalization .ipynb
These notebooks that act a main function call and includes the following steps

1. Google drive set up to store your code and link the drive to the Notebook.
2. Define the transformation for test and train dataset. 
3. Download the CIFAR10 data from the datasets package and apply the transformation.
 The image size is 32 X 32 and we have 60K images in train and 10K images in the test set

4. Plot a sample set of images using the functions rom utils library
![smaple_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/Sample%20Images.png)

6. Run the model
With a batch size of 512 we are running 15 epochs.
Optimasation method is Stochastic Gradient Decent and the Loss function is  negative log likelihood loss
The final accuracy at 15th  epoch is >70% for train and test set.


### Solution: Target Result and Analysis ✌✌️
### Step 1
### 🎯 Target
1. Create a base model with convolutions of stride 2 instead of max pooling in all the 3 convolution blocks
2. Experiment with 15 epochs and observe the accuracies

### 💪 Result
1. Parameters : 58314
2. Best Train Accuracy: 82.05
3. Best Test Accuracy: 79.03

### 👀 Analysis
1. The test accuracy is fluctuating till 15th epoch. The gap between the test and train accuracy is also fluctuating.
2. The receptive field at the final layer is 47
3. Had to use a couple of 4X4 Convolutions to have channel sizes of decimals like (16.5)
4. The accuracy was not consistently above 85% for the last few epochs


**Receptive feild calculation**


**Model Summary**


**Last few epochs**


**Misclassified images**


**Link to the Notebook**



### Step 2
### 🎯 Target
1. Add a dilated convolution instead of a 3 X 3 convolution of stride 2
2. Experiment with 20 epochs and observe the accuracies
3. Add a depthwise seperable convolution
4. Add Augmentation
5. Run for 50 epochs

### 💪 Result
1. Parameters : 49746
2. Best Train Accuracy: 81.14
3. Best Test Accuracy: 80.44

### 👀 Analysis
1. We reached the 85% accuracy at 50 the epoch.
2. USe albumenttaion for augmentation
3. Re look at the architecture again, avoid 1X1 to reduce the channels

**Receptive feild calculation**
![receptive_feild](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/Receptive%20Feild%20Calculations.png)

**Model Summary**
![bn_model](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/model.png)

**Last few epochs**
![bn_epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/epochs.png)

**Misclassified images**
![bn_misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/missclassified%20images.png)

**Link to the Notebook**
[Model with Augmentation](https://github.com/prarthananbhat/ERA/blob/master/Session_9/S9_model_1.ipynb)



