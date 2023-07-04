from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616)),    
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.RandomCrop(size=[32,32], padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
    ])

train_transforms_A = A.Compose([
    A.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
    A.ShiftScaleRotate(),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes = 1, 
                    max_height=16, 
                    max_width=16, 
                    min_holes = 1, 
                    min_height=16, 
                    min_width=16),
    ToTensorV2()
    
])

test_transforms_A = A.Compose([
    A.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
    ToTensorV2()
])
