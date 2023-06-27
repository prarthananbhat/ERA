## Assignment from Session 8 - Batch Normalization & Regularization

### Objective: 
1. Change the dataset to CIFAR10
2. Make this network: C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
   a. Keep the parameter count less than 50000
   b. Try and add one layer to another
   c. Max Epochs is 20
3. Make 3 versions of the above code (in each case achieve above 70% accuracy):
   a. Network with Group Normalization
   b. Network with Layer Normalization
   c. Network with Batch Normalization
4. Share these details
   a. Training accuracy for 3 models
   b. Test accuracy for 3 models
   c. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images. 

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
[Model with Batch Normalization Notebook](https://github.com/prarthananbhat/ERA/blob/master/Session_8/S8%20-%20Batch%20Normalization%20.ipynb)


### Step 3
### Target
1. Reduce the number of parameters
2. Use Batch Normalisation and drop out

### Result
1. Parameters : 6402
2. Best Train Accuracy: 81.14
3. Best Test Accuracy: 80.44

### Analysis
1. The test accuracy is slightly lower than the train accuracy, We can reduce this gap using regularization on Augmentation
2. The parameters are now less than 50k and achieved an accuracy of 80%
3. We reached a receptive felid of 72

**Receptive feild calculation**
![calc](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step)

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step_3/step_3_model.png)

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step_3/step_3_epochs.png)


### Step 4
### Target
1. Use image augmentation
2. Use the exact same model liske step 3

### Result
1. Parameters : 6402
2. Best Train Accuracy: 97.68
3. Best Test Accuracy: 97.84

### Analysis
1. We have still not reached the required accuracy.
2. We have to change the number of kernels at different steps probably.

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step%204/step_4_epochs.png)


### Step 5
### Target
1. Increase Model Capacity
2. Run up-till 20 epochs to see if we achieve result after 15 epochs

### Result
Parameters : 7738
Best Train Accuracy: 99.20
Best Test Accuracy: 99.53

### Analysis
1. We achieved the desired result at 15th epoch.
2. Lets try increasing the lr slightly to achieve this faster

**Receptive feild calculation**
![calc](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step_5/step_5_calculations.png)

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step_5/step_5_model.png)

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/step_5/step_5_epochs.png)

### Step 6
### Target
1. Increase the lr from 0.01 to 0.015

### Result
1. Parameters : 7738
2. Best Train Accuracy: 99.12
3. Best Test Accuracy: 99.53
   
### Analysis
1. We achieved the result constantly after 12th epoch
2. No image augmentation
3. No lr scheduler

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_7/misc/Screenshot%202023-06-15%20at%2011.37.22%20PM.png)

