## Assignement from Session 4

### Building the First Neural Networks

#### Objective: Identify the digital from the MNIST data set using a Convolutional Neural Network and organise the code into 
```
.
├── README.md
├── S5.ipynb
├── model.py
└── utils.py
```


#### models.py
Here we define the architecture of our model. 
We Chose to have 
1. 4 convolutional layers, with max pool after 2 layers.
2. 2 fully connected layers at the end.
3. This model has 593200 parameters

Below is a snapshot of the model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
----------------------------------------------------------------

Total params: 593,200                                           
Trainable params: 593,200
Non-trainable params: 0
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94

#### Utils.py
This files stores the utility function. We have 2 plotting functions in this file.
plot_samples function plots 12 images from the data loader
Plot accuracy metrics plot the training and testing loss and accuracy curves for the complete run (all epochs)


#### S5.ipynb 
is a notebook that acts a main function call and includes the following steps

1. Google drive set up to store your code and link the drive to the Notebook.
2. Define the transformation for test and train dataset. 
3. Download the MNIST data from the datasets package and apply the transformation.
 The image size is 28 X 28 and we have 60K images in train and 10K images in the test set

4. Plot a sample set of images using the functions rom utils library
![smaple_images](https://github.com/prarthananbhat/ERA/blob/master/Session_5/misc/download.png)

6. Run the model
With a batch size of 512 we are running 20 epochs.
Optimasation method is Stochastic Gradient Decent and the Loss function is Cross entropy
We have also used a lr scheduler to change our learning rate after 15 epochs.
The final accuracy at 20th epoch is 99.21% for train and 99.19$ for test set.
![smaple_images](https://github.com/prarthananbhat/ERA/blob/master/Session_5/misc/accuracy_metrics.png)





