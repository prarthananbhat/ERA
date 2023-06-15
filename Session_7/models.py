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



