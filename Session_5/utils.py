import matplotlib.pyplot as plt

def plot_samples(batch_data,batch_label):
  fig = plt.figure()

  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])
  plt.show()
  