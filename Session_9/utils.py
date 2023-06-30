import matplotlib.pyplot as plt

def plot_samples(batch_data,batch_label):
  fig = plt.figure(figsize = (8,8))

  for i in range(64):
    plt.subplot(8,8,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])
  plt.show()

def plot_accuracy_metrics(train_losses,train_acc,test_losses,test_acc, epoch):

  fig, ax = plt.subplots(1,2,figsize=(12,5))
  ax[0].plot(train_losses,label="Train Loss")
  ax[1].plot(train_acc,label="Train Accuracy")
  ax[0].plot(test_losses, label="Test Loss")
  ax[1].plot(test_acc,label="Test Accuracy")
  ax[1].legend(loc='upper left')
  ax[0].legend(loc='upper left')
  ax[1].set_title(f"Accuracy Curve for {epoch} Epochs")
  ax[0].set_title(f"Loss Curve for {epoch} Epochs")
  ax[1].set_ylabel('Accuracy')
  ax[0].set_ylabel('Loss')
  ax[0].set_xlabel('Epochs')
  ax[1].set_xlabel('Epochs')
  plt.show()


def plot_samples_cifar10(batch_data,batch_label,classes):
  fig = plt.figure(figsize = (8,8))

  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].permute(1,2,0))
    plt.title(classes[batch_label[i].item()])
    plt.xticks([])
    plt.yticks([])
  plt.show()


def plot_misclassified_samples(batch_data,batch_label, pred_label,classes):
  fig = plt.figure(figsize = (8,8))

  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].permute(1,2,0))
    plt.title(f"ActuaL:{classes[batch_label[i]]}, \nPredicted : {classes[pred_label[i]]}")
    plt.xticks([])
    plt.yticks([])
  plt.show()
  


def plot_samples_cifar10_1(batch_data,batch_label,classes,unnorm):
  fig = plt.figure(figsize = (8,8))

  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(unnorm(batch_data[i]).permute(1,2,0))
    plt.title(classes[batch_label[i].item()])
    plt.xticks([])
    plt.yticks([])
  plt.show()


def plot_misclassified_samples_1(batch_data,batch_label, pred_label,classes,unnorm):
  fig = plt.figure(figsize = (8,8))

  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(unnorm(batch_data[i]).permute(1,2,0))
    plt.title(f"ActuaL:{classes[batch_label[i]]}, \nPredicted : {classes[pred_label[i]]}")
    plt.xticks([])
    plt.yticks([])
  plt.show()