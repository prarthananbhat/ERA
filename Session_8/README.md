## 🤖 Assignment from Session 8 - Batch Normalization & Regularization

### Objectiv 🏆 
1. Change the dataset to CIFAR10.
2. Make this network: C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10.
   1. Keep the parameter count less than 50000
   2. Try and add one layer to another
   3. Max Epochs is 20
3. Make 3 versions of the above code (in each case achieve above 70% accuracy):
   1. Network with Group Normalization
   2. Network with Layer Normalization
   3. Network with Batch Normalization
4. Share these details
   1. Training accuracy for 3 models
   2. Test accuracy for 3 models
   3. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images. 

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
1. Change the dataset to CIFAR from MNIST, remove the augmentations on the images
2. Create the First network with Convolution, max pooling and CAP layers. We are not condsidering parameter count or accuracies for the moment
3. Plot the accuracy Metrics 
4. Create a table to calculate receptive field, number of parameters
6. Reach a receptive field of atleast 32
7. create a function to plot the missclassified images.
8. The number of paramters, it should be less than 50K
2. Use batch noramlization after all 3X3 Colvolutions
3. Achieve an accuracy for 70 or more within 20 epochs.

### 💪 Result
1. Parameters : 49746
2. Best Train Accuracy: 84.14
3. Best Test Accuracy: 74.38

### 👀 Analysis
1. The model is overfitting, after 15th epoch the losses are increasing
2. The target of 70% is acheieved within 20 epochs
3. We reached a receptive felid of 72

**Receptive feild calculation**
![step_2_calculations](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/batch_normalization/BN_receptive%20feild%20calculation.png)

**Model Summary**
![bn_model](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/batch_normalization/bn_model.png)

**Last few epochs**
![bn_epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/batch_normalization/bn_epochs.png)

**Misclassified images**
![bn_misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/batch_normalization/bn_misclassified%20images.png)

**Link to the Notebook**
[Model with Batch Normalization](https://github.com/prarthananbhat/ERA/blob/master/Session_8/S8%20-%20Batch%20Normalization%20.ipynb)


### Step 2
### 🎯 Target
1. Change the batch normalization to group normalization
2. Use the nn.GroupNorm(2, 16), Which creates 2 groups from 16 kerenels. If our batch size is 32, the we will have 32(images) * 2(groups) * 2(mean and sd) = 128 parameters

### 💪 Result
1. Parameters : 49746 (ideally parameters should have changed)
2. Best Train Accuracy: 83.57
3. Best Test Accuracy: 77.31

### 👀 Analysis
1. The model is overfitting too, after 15th epoch the losses are increasing
2. The target of 70% is acheieved within 20 epochs
3. We reached a receptive felid of 72

**Receptive feild calculation**
![gn_calculations](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/group%20Normailization/gn_receptive%20feild%20calculation.png)

**Model Summary**
![gn_model](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/group%20Normailization/gn_model.png)

**Last few epochs**
![gn_epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/group%20Normailization/gn_epochs.png)

**Misclassified images**
![gn_misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/group%20Normailization/gn_missclassified_images.png)

**Link to the Notebook**
[Model with Group Normalization](https://github.com/prarthananbhat/ERA/blob/master/Session_8/S8%20-%20Group%20Normalization%20.ipynb)


### Step 3
### 🎯 Target
1. Change the group normalization to Layer normalization
2. Use the nn.GroupNorm(1, 16), Which creates 1 groups from 16 kerenels. If our batch size is 32, the we will have 32(images) * 1(groups) * 2(mean and sd) = 64 parameters

### 💪 Result
1. Parameters : 49746 (ideally parameters should have changed)
2. Best Train Accuracy: 81.58
3. Best Test Accuracy: 73.53

### 👀 Analysis
1. The model is overfitting too, after 15th epoch the losses are increasing
2. The target of 70% is acheieved within 20 epochs
3. We reached a receptive felid of 72


**Receptive feild calculation**
![ln alculations](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/layer%20Normalization/ln_receptive%20feild%20calculations.png)

**Model Summary**
![ln_model](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/layer%20Normalization/Step%202%20Model.png)

**Last few epochs**
![ln_epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/layer%20Normalization/Step%202%20epochs.png)

**Misclassified images**
![ln_misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/layer%20Normalization/Step%202%20missclassified%20images.png)

**Link to the Notebook**
[Model with Layer Normalization](https://github.com/prarthananbhat/ERA/blob/master/Session_8/S8%20-%20Layer%20Normalization%20.ipynb)


