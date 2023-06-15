## Assignment from Session 7

### In-Depth Coding Practice

#### Objective: 
1.  The model Accuracy should be 99.4%  **(this must be consistently shown in your last few epochs, and not a one-time achievement)**
2. Less than or equal to 15 Epochs
3. Less than 8000 Parameters
4. Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.
5.  Do this in exactly 3 steps
6.  Each File must have a "target, result, analysis" TEXT block (either at the start or the end)
7.  Write why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct.
8. Keep Receptive field calculations handy for each of your models.

#### Solution
#### Step 1
#### Target
1. Modularise the model
2. Create the First network with Convolution, max pooling and fc layers.
3. Plot the accuracy Metrics 
4. Create a table to calculate receptive field, number of parameters
5. ![step_1_calculations](https://drive.google.com/file/d/12uQHWGYf5ZV8nwkltVwIDw3MdCLRTwgO/view?usp=drive_link)
6. Reach a receptive field of 16 to 20

#### Result
Parameters : 428810
Best Train Accuracy: 99.43
Best Test Accuracy: 99.52

#### Analysis
1. The model is not overfitting till 15th epoch. The test accuracy is higher than the train accuracy.
2. The parameters are very high for a data set like MNIST.
3. The architecture has only 3X3 convolutions, Max pooling and FC.


#### Step 2
#### Target
1. Increase the receptive field
2. Reduce the number of parameters
3. Use Average pooling at the end
4. Use 1X1 to reduce the dimensions

#### Result
Parameters : 11226
Best Train Accuracy: 96.79
Best Test Accuracy: 96.73

#### Analysis
1. The model is not overfitting till 15th epoch. The test accuracy is similar to train accuracy
2. The parameters are still higher than 8K
3. We reached a receptive felid of 28
4. We did not achieve the required accuracy of 99.4


#### Step 3
#### Target
1. Reduce the number of parameters
2. Use Batch Normalisation and drop out

#### Result
Parameters : 6402
Best Train Accuracy: 98.56
Best Test Accuracy: 98.72

#### Analysis
1. The model is not overfitting till 15th epoch. The test accuracy is higher than the train accuracy.
2. The parameters are now less than 8k
3. We reached a receptive felid of 28
4. We have not reached the required accuracy. 

#### Step 4
#### Target
1. Use image augmentation
2. 
#### Result
Parameters : 6402
Best Train Accuracy: 97.67
Best Test Accuracy: 97.87

#### Analysis
1. We have still not reached the required accuracy.
2. We have to change the number of kernels at different steps probably.
3. 


#### Step 5
#### Target
1. Increase Model Capacity
2. Run up-till 20 epochs to see if we achieve result after 15 epochs

#### Result
Parameters : 7738
Best Train Accuracy: 99.16
Best Test Accuracy: 99.53

#### Analysis
1. We achieved the desired result at 17th epoch.
2. Lets try increasing the lr slightly to achieve this faster

#### Step 6
#### Target
1. Increase the lr from 0.01 to 0.015

#### Result
Parameters : 7738
Best Train Accuracy: 99.12
Best Test Accuracy: 99.53

#### Analysis
1. We achieved the result constantly after 12th epoch
2. No image augmentation
3. No lr scheduler