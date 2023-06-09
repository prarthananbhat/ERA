## S6 - Assignment

Write a network on **MNIST** Data such that
1. you achieve **99.4%** validation accuracy
2. Model should have Less than **20k Parameters**
3. You can use anything from above you want
4. Less than **20 Epochs**


### Model architecture

I have used a **block** architecture with 3 blocks to achieve the above results. The input images are grayscale with a size of 28x28 pixels.
1. Every block will exapnd the number of channels and then reduce it. To **expand** the number of channels we use a **3 X 3 Kernel** and to **reduce** it we use **1 X 1 Kernal**. In the below architecture, I am increasing the number of channels to 32 then reducing it to 8. 
2. At the end of the block **Batch Normalization** is used to normalise the activations in each batch and solve the issue of covariate shift. 
3. Dropout with a rate of 0.20 is applied at the end of the block to randomly deactivate some neurons during training, which helps prevent overfitting.
4. The channel size is maintained at 28 X 28 as padding is introduced. Max pooling with a pool size of 2x2 is performed to downsample the spatial dimensions and reduce computational complexity. This reduces the channel size by half at the end of each block.
5. The last block is a little different from the first 3 blocks as it feeds to the final softmax. **The 1 X 1 reduces the number of channels to 10**. After this my channels will be 3 X 3 X 10. 
6. Now I use and **average pool** of **size 3** to **reduce my channel size to 1**. My final layer produces a 1 X 1 X 10 which is an input to a softmax. Avergae pool is itentionally put after a drop out as we do not want to drop any infomration in the final output. 

7. By introducing 1 X 1 and Avergae pool at the lasyer we can avoid using a fully connected layer at the end.

***The total number of paaremters in this model are 18,666***



----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
            Conv2d-4           [-1, 16, 28, 28]           1,168
              ReLU-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
            Conv2d-7           [-1, 32, 28, 28]           4,640
              ReLU-8           [-1, 32, 28, 28]               0
       BatchNorm2d-9           [-1, 32, 28, 28]              64
        MaxPool2d-10           [-1, 32, 14, 14]               0
           Conv2d-11            [-1, 8, 14, 14]             264
          Dropout-12            [-1, 8, 14, 14]               0
           Conv2d-13           [-1, 16, 14, 14]           1,168
             ReLU-14           [-1, 16, 14, 14]               0
      BatchNorm2d-15           [-1, 16, 14, 14]              32
           Conv2d-16           [-1, 32, 14, 14]           4,640
             ReLU-17           [-1, 32, 14, 14]               0
      BatchNorm2d-18           [-1, 32, 14, 14]              64
        MaxPool2d-19             [-1, 32, 7, 7]               0
           Conv2d-20              [-1, 8, 7, 7]             264
          Dropout-21              [-1, 8, 7, 7]               0
           Conv2d-22             [-1, 16, 7, 7]           1,168
             ReLU-23             [-1, 16, 7, 7]               0
      BatchNorm2d-24             [-1, 16, 7, 7]              32
           Conv2d-25             [-1, 32, 7, 7]           4,640
             ReLU-26             [-1, 32, 7, 7]               0
      BatchNorm2d-27             [-1, 32, 7, 7]              64
        MaxPool2d-28             [-1, 32, 3, 3]               0
           Conv2d-29             [-1, 10, 3, 3]             330
          Dropout-30             [-1, 10, 3, 3]               0
        AvgPool2d-31             [-1, 10, 1, 1]               0
----------------------------------------------------------------
Total params: 18,666

Trainable params: 18,666 

Non-trainable params: 0

Input size (MB): 0.00

Forward/backward pass size (MB): 1.37

Params size (MB): 0.07

Estimated Total Size (MB): 1.44



