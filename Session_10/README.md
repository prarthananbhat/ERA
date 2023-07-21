## 🤖 Assignment from Session 10 - Residual Connections in CNNs and One Cycle Policy!

### Objectiv 🏆 
1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:
   1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
   2. Layer1 -
   3. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
   4. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
   5. Add(X, R1)
   6. Layer 2 -
   7. Conv 3x3 [256k]
   8. MaxPooling2D
   9. BN
   10. ReLU
   11. Layer 3 -
   12. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
   13. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
   14. Add(X, R2)
   15. MaxPooling with Kernel Size 4
   16. FC Layer
   17. SoftMax
2. Uses One Cycle Policy such that:
3. Total Epochs = 24
4. Max at Epoch = 5
5. LRMIN = FIND
6. LRMAX = FIND
7. NO Annihilation
8. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
9. Batch size = 512
10. Use ADAM, and CrossEntropyLoss
11. Target Accuracy: 90%

### Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Here are the classes in the dataset, as well as 10 random images from each:

![sample_images](https://github.com/prarthananbhat/ERA/blob/master/Session_8/misc/sample_images_downloaded.png)

### Code Base
```
.
|-- README.md
|-- S10_model.ipynb
|-- test.py
|-- train.py
|-- models.py
|-- utils.py
|-- dataset.py
|-- transform.py
|-- misc
```

#### models.py
This file holds all the model definitions (network architecture). 2 models were added as part of this assignm,ent. 
1. s10_model - *RESNET model with 6,573,120 paraneters*

**Receptive feild calculation**
![receptive_feild]()

**Model Summary**
![model](https://github.com/prarthananbhat/ERA/blob/master/Session_10/misc/model.png)

#### utils.py
all the plotting functions required for accuracy_metrics, sample images plot and misclassified images is avaible in this file.

#### dataset.py
Two classes are defined here,
1. **Unormalise Class** to denormalise the images using standard deviation and mean
2. **cifardataset class** to create a custom dataset with albumentation transforms.

#### transform.py
We have two set of transforms in this file, One from the pytorch transforms library and other from the albumentations Library

#### train.py
Model gets trained for everybatch in the file. We have also added the scheduler.step() for oncycle lr in the same code. Train losses and accuracies are calculated here.

#### test.py
Predictions from the trained model for every batch happens here. Test losses and accuracies are caluclted.

#### Notebooks S10_model.ipynb
This notebook that act a main function call and includes the following steps

1. Google drive set up to store your code and link the drive to the Notebook.
2. Define the transformation for test and train dataset. 
3. Download the CIFAR10 data from the datasets package and apply the transformation.
 The image size is 32 X 32 and we have 60K images in train and 10K images in the test set

4. Plot a sample set of images using the functions rom utils library
![smaple_images](https://github.com/prarthananbhat/ERA/blob/master/Session_10/misc/sample_images.png)

6. Run the model
With a batch size of 512 we are running 15 epochs.
Optimasation method is *ADAM* and the Loss function is *Cross Entropy Loss*


### Solution: Target Result and Analysis ✌✌️
### 🎯 Target
1. Create a RESNET Model
2. Use LR Finder to find the optimal LR (*a learning rate where there is a maximum loss drop*)
3. Use One Cycle LR poilicy to increase the learning rate to maximum of 100 times of your initial learning rate and then reduce it.
4. Run for 24 Epochs

### 💪 Result
1. Parameters : 6,573,120
2. Best Train Accuracy: 90.63
3. Best Test Accuracy: 90.27

### 👀 Analysis
1. The losses for both train and test were fluctuating a lot because of oncycle_lr.
2. We reached the target accuracy of 90% in the 24th epoch.
3. The best lr found by the fr finder was 0.0005

**Last few epochs**
![epochs](https://github.com/prarthananbhat/ERA/blob/master/Session_10/misc/epochs.png)

**Misclassified images**
![misclassified_images](https://github.com/prarthananbhat/ERA/blob/master/Session_10/misc/missclassified%20images.png)

**Loss Curves**
![loss_curves](https://github.com/prarthananbhat/ERA/blob/master/Session_10/misc/loss%20curve.png)

**Link to the Notebook**
[Base model](https://github.com/prarthananbhat/ERA/blob/master/Session_10/S10_model.ipynb)





