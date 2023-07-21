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
|-- S9 - base model.ipynb
|-- S9_model_1.ipynb
|-- S9_model_2.ipynb
|-- models.py
|-- utils.py
|-- dataset.py
|-- transform.py
|-- misc
```

#### models.py
This file holds all the model definitions (network architecture). 2 models were added as part of this assignm,ent. 
1. s9_base_model - *Inital model with strided convolution*
2. s9_model_1 - *Model with Depthwise convolution and dialated convolution*
3. s9_model_2 - *Model with Depthwise convolution and dialated convolution and augmentations from albumentation*


#### utils.py
This files stores the utility function. 2 additional plotting functions are added
**plot_samples_cifar** function plots 12 images from the data loader
**plot_misclassified_images** plots the 25 images that are wrongly specified with titles like *Predicted : Cat, Actual Dog*

#### dataset.py
Two classes are defined here,
1. **Unormalise Class** to denormalise the images using standard deviation and mean
2. **cifardataset class** to create a custom dataset with albumentation transforms.

#### transform.py
We have two set of transforms in this file, One from the pytorch transforms library and other from the albumentations Library

#### Notebooks S9 - base model.ipynb .ipynb, S9_model_1.ipynb, S9_model_2.ipynb
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


### Solution: Target Result and Analysis ✌✌️
### Step 1
### 🎯 Target
1. Create a base model with convolutions of stride 2 instead of max pooling in all the 3 convolution blocks
2. Experiment with 50 epochs and observe the accuracies
3. Parameters should be less than 200K

### 💪 Result
1. Parameters : 111978
2. Best Train Accuracy: 98.04
3. Best Test Accuracy: 77.4

### 👀 Analysis
1. Not a great model, Overfits like crazy 
2. The receptive field at the final layer is 47
3. Had to use a couple of 4X4 Convolutions to have channel sizes of decimals like (16.5)
4. After 35th epoch we see that test accuracies were decresing

**Receptive feild calculation**
![receptive_feild](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%201/receptive%20feild%20calculations.png)

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%201/base_model.png)

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%201/base_model_epochs.png)

**Misclassified images**
![misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%201/missclassified%20images.png)

**Loss Curves**
![loss_curves](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%201/base_model_loss_curves.png)

**Link to the Notebook**
[Base model](https://github.com/prarthananbhat/ERA/blob/master/Session_9/S9%20-%20base%20model.ipynb)



### Step 2
### 🎯 Target
1. Increase capacity in the model
2. Add a dilated convolution instead of a 3 X 3 convolution of stride 2
3. Experiment with 50 epochs and observe the accuracies
4. Add a depthwise seperable convolution
5. Add Augmentation from pytorch library
   1.Random Rotation, Random Crop, Horizontal flip and Random Affine 
6. Run for 50 epochs

### 💪 Result
1. Parameters : 191898
2. Best Train Accuracy: 81.84
3. Best Test Accuracy: 84.63

### 👀 Analysis
1. We reached the 84.63% accuracy at 50 the epoch.
2. Thr overfitting was reduced after adding augmentation
3. The epochs were quite slow, due to trasformations

**Receptive feild calculation**
![receptive_feild](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/Receptive%20Feild%20Calculations.png)

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/model.png)

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/epochs.png)

**Misclassified images**
![misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/missclassified_images.png)

**Loss Curves**
![loss_curves](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%202/loss_curves.png)

**Link to the Notebook**
[Model with Augmentation](https://github.com/prarthananbhat/ERA/blob/master/Session_9/S9_model_1.ipynb)


### Step 3
### 🎯 Target
1. Use Augmentation from albumentation package
   1. ShiftScaleRotate
   2. HorizontalFlip
   3. CoarseDropout   

### 💪 Result
1. Parameters : 191898
2. Best Train Accuracy: 78.17
3. Best Test Accuracy: 83.02

### 👀 Analysis
1. We reached the 83% accuracy at 50 the epoch.
2. Thr overfitting was reduced after adding augmentation
3. The albumentation transforms were faster compared to pytorch transfomrations

**Transformed Images**
![transformed_images](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%203/sample_transformed_images.png)

**Receptive feild calculation**
![receptive_feild](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%203/receptive%20feild%20calculations.png)

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%203/model.png)

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%203/epochs.png)

**Misclassified images**
![misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_9/misc/Step%203/missclassified_images.png)

**Link to the Notebook**
[Model with Augmentation](https://github.com/prarthananbhat/ERA/blob/master/Session_9/S9_model_2.ipynb)



