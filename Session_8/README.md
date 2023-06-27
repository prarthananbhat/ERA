## Assignment from Session 8 - Batch Normalization & Regularization

### Objective: 
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

### Solution
### Step 1
### Target
1. Change the dataset to CIFAR from MNIST, remove the augmentations on the images
2. Create the First network with Convolution, max pooling and CAP layers. We are not condsidering parameter count or accuracies for the moment
3. Plot the accuracy Metrics 
4. Create a table to calculate receptive field, number of parameters
6. Reach a receptive field of atleast 32
7. create a function to plot the missclassified images.

### Result
1. Parameters : 58314
2. Best Train Accuracy: 82.05
3. Best Test Accuracy: 79.03

### Analysis
1. We have got the required test accuracioes
2. Parameters are little higher than expected
3. The architecture has only 3X3 ,1X1 convolutions, Max pooling and GAP.

**Receptive feild calculation**
![step_1_calculations](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/Step%201/Step%201%20Receptive%20Feild%20Caluculation.png)

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/Step%201/Step%201%20Model.png)

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/Step%201/Step%201%20epochs.png)

**Misclassified images**
![misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/Step%201/misclassified_images.png)

**Link to the Notebook**
[Initial Notebook](https://github.com/prarthananbhat/ERA/blob/master/Session_8/S8%20-%20Batch%20Normalization%20.ipynb)


### Step 2
### Target
1. Reduce the number of paramters, it should be less than 50K
2. Use batch noramlization after all 3X3 Colvolutions
3. Achieve an accuracy for 70 or more within 15 epochs.

### Result
1. Parameters : 49746
2. Best Train Accuracy: 81.14
3. Best Test Accuracy: 80.44

### Analysis
1. The test accuracy is slightly lower than the train accuracy, We can reduce this gap using regularization on Augmentation
2. The parameters are now less than 50k and achieved an accuracy of 80%
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


### Step 3
### Target
1. Change the batch normalization to group normalization
2. Use the nn.GroupNorm(2, 16), Which creates 2 groups from 16 kerenels. If our batch size is 32, the we will have 32(images) * 2(groups) * 2(mean and sd) = 128 parameters

### Result
1. Parameters : 49746 (ideally parameters should have changed)
2. Best Train Accuracy: 76.29
3. Best Test Accuracy: 77.59

### Analysis
1. The test accuracy is similar to train or slightly higher till 15th epoch
2. The parameters are now less than 50k and achieved an accuracy of 77.6%
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


### Step 4
### Target
1. Change the group normalization to Layer normalization
2. Use the nn.GroupNorm(1, 16), Which creates 1 groups from 16 kerenels. If our batch size is 32, the we will have 32(images) * 1(groups) * 2(mean and sd) = 64 parameters

### Result
1. Parameters : 49746 (ideally parameters should have changed)
2. Best Train Accuracy: 76.9
3. Best Test Accuracy: 78.1

### Analysis
1. The test accuracy is similar to train or slightly higher till 15th epoch
2. The parameters are now less than 50k and achieved an accuracy of 78.1%
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


