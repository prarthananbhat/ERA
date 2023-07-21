import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.fc1 = nn.Linear(4*4*256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)
        # x = F.relu(self.conv6(x), 2)
        x = x.view(-1, 4*4*256)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)



class model_2(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(model_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv7 = nn.Conv2d(16, 10, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(7)



    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(self.conv2(x), 2)
        x = F.relu(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)

        x = F.relu(self.conv5(x), 2)
        x = F.relu(self.conv6(x), 2)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class model_3(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(model_3, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(1, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(16, 8, 1),
                  nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(8, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 10, 1),
                  nn.AvgPool2d(8, 8),
              )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class model_4(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(model_4, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(1, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(16, 8, 1),
                  nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(8, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(16, 8, 1),
                  nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(8, 16, 3, padding=0),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 10, 1),
                  nn.AvgPool2d(2,2),
              )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class cifar_model(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(cifar_model, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),                 
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.AvgPool2d(8,8),
                  nn.Conv2d(32, 10, 1),

              )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class cifar_model_gn(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(cifar_model_gn, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 16, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 16),
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 24, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 24),
                  nn.Conv2d(24, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(4, 32),
                  nn.AvgPool2d(8,8),
                  nn.Conv2d(32, 10, 1),

              )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class cifar_model_ln(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(cifar_model_ln, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 16, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 16),
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 24, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 24),
                  nn.Conv2d(24, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.AvgPool2d(8,8),
                  nn.Conv2d(32, 10, 1),

              )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class cifar_model_bn(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(cifar_model_bn, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 16, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 24, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(24),
                  nn.Conv2d(24, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),                 
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.AvgPool2d(8,8),
                  nn.Conv2d(32, 10, 1),

              )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class cifar_model_ln(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(cifar_model_ln, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 16, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 16),
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 24, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 24),
                  nn.Conv2d(24, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 16, 1),
                  nn.MaxPool2d(2, 2),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.GroupNorm(1, 32),
                  nn.AvgPool2d(8,8),
                  nn.Conv2d(32, 10, 1),

              )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)




class s9_base_model(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(s9_base_model, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 16, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 4, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1, stride = 2),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),    
                  nn.Conv2d(32, 16, 1),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 4, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),                 
                  nn.Conv2d(32, 32, 3, padding=1, stride = 2),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),                 
                  nn.Conv2d(32, 16, 1),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 4, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1, stride = 2),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.AvgPool2d(4,4),
                  nn.Conv2d(32, 10, 1),

              )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class s9_model_1(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(s9_model_1, self).__init__()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 4, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32,32 , 3, padding=1, stride = 1, dilation = 1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 16, 1),
                  # nn.Dropout(0.10)
              )
        self.conv2 = nn.Sequential(
                  nn.Conv2d(16, 64, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, 3, padding=1, stride = 2, dilation = 1),
                  nn.ReLU(),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 16, 1),
                  # nn.Dropout(0.10)
              )
        self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 32, 3, padding=1, stride = 2, dilation = 1),
                  nn.ReLU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32),
                  nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.AvgPool2d(5,5),
                  nn.Conv2d(16, 10, 1),
              )

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


# ********************************************S10 MODEL***************************************************************
dropout_value = 0.1

class s10_model(nn.Module):
    def __init__(self):
        super(s10_model, self).__init__()

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
            )
         #Layer1 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] (Conv-BN-ReLU-Conv-BN-ReLU)
# Add(X, R1)
        self.x1 = nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            )
        self.R1 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
            )

         #Layer 2 -
# Conv 3x3 [256k]
# MaxPooling2D
# BN
# ReLU
        self.layer2 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            )

#         Layer 3 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
# R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
# Add(X, R2)
        self.x2 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            )

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
            )


        self.pool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(in_features = 512, out_features = 10, bias=False)




    def forward(self, x):

        preplayer = self.preplayer(x) # i/p= 32 o/p=32 Rf = 6
        x1 = self.x1(preplayer)
        R1 = self.R1(x1)
        layer1 = x1+R1
        layer2 = self.layer2(layer1)
        x2 = self.x2(layer2)
        R2 = self.R2(x2)
        layer3 = R2+x2
        maxpool = self.pool(layer3)

       #print(maxpool.shape)

        # fc = self.fc(x)

        #print(fc.shape)
        #print(maxpool.shape)
        x = maxpool.view(maxpool.size(0),-1)
        #print(x.shape)
        #print(x.shape)
        fc = self.fc(x)

        return F.log_softmax(fc.view(-1,10), dim=-1)





