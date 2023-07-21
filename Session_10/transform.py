from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# train_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616)),    
#     transforms.RandomRotation((-15., 15.), fill=0),
#     transforms.RandomCrop(size=[32,32], padding=4),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#     ])

# test_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
#     ])

# train_transforms_A = A.Compose([
#     A.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
#     A.ShiftScaleRotate(),
#     A.HorizontalFlip(p=0.5),
#     A.CoarseDropout(max_holes = 1, 
#                     max_height=16, 
#                     max_width=16, 
#                     min_holes = 1, 
#                     min_height=16, 
#                     min_width=16),
#     ToTensorV2()
    
# ])

# test_transforms_A = A.Compose([
#     A.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
#     ToTensorV2()
# ])

means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

train_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=10, max_width=10, min_holes=1, min_height=10, min_width=10, fill_value=means),



        #A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),
    ]
)



