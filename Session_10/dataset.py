from torchvision import datasets

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class cifarDataset(datasets.CIFAR10):
  def __init__(self, root = "../data",train = True,transform = None,target_transform = None,
          download = True):
      super().__init__(root = root, train = train, download = download, transform=transform, target_transform=target_transform)

  def __getitem__(self, index: int):
          """
          Args:
              index (int): Index

          Returns:
              tuple: (image, target) where target is index of the target class.
          """
          img, target = self.data[index], self.targets[index]
          if self.transform is not None:
            transformed = self.transform(image = img)
            image = transformed["image"]
          return image, target
