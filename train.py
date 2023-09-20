import datetime
import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from model import Model4layerMLP
from torch.utils.data import DataLoader
from torchsummary import summary
import argparse



########################################### single pic test ################################################3
# X = train_set.data[2]
# X = X.reshape(1,784)
# print(X)
# X = X.type(torch.float32)
# with torch.inference_mode(): # <- context manager, turns off gradient tracking and so to save time, memory
#   y_preds = model(X)
# print(y_preds)
# X = X.reshape(28,28)
# y_preds = y_preds.reshape(28,28)
# plt.imshow(X, cmap="gray")
# plt.show()
# plt.imshow(y_preds,cmap="gray")
# plt.show()

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, plot_loss_path, save_model_path):
  print("training...")
  model.to(device)
  model.train()
  losses_train = []

  for epoch in range(1, n_epochs+1):
    print('epoch', epoch)
    loss_train = 0.0
    for imgs in train_loader:
      imgs = imgs.to(device=device)
      imgs = imgs.reshape(1,784)
      outputs = model(imgs)
      loss = loss_fn(outputs, imgs)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()

    # if epoch % 10 == 0:
    #   with torch.inference_mode():
    #     f = plt.figure()
    #     f.add_subplot(1, 2, 1)
    #     t_img = train_loader[39].reshape(28, 28)
    #     plt.imshow(t_img, cmap = 'gray')
    #     f.add_subplot(1, 2, 2)
    #     f_img = model(train_loader[39].reshape(1,784)).reshape(28, 28)
    #     plt.imshow(f_img, cmap='gray')
    #     plt.show()

    scheduler.step(loss_train)

    losses_train += [loss_train/len(train_loader)]

    print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now()
                                               , epoch, loss_train/len(train_loader)))

  plt.plot(losses_train)
  plt.savefig(plot_loss_path)
  plt.show()
  torch.save(model.state_dict(), save_model_path)
  summary(model, (1, 784))


def test(train_loader, model, model_param,device):
  model.to(device)
  model.load_state_dict(torch.load(model_param))
  for i in range(len(train_loader[:2047])):
    if i % 900 == 0:
      with torch.inference_mode():
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        t_img = train_loader[i].reshape(28, 28)
        plt.imshow(t_img, cmap='gray')
        f.add_subplot(1, 2, 2)
        f_img = model(train_loader[i].reshape(1, 784)).reshape(28, 28)
        plt.imshow(f_img, cmap='gray')
        plt.show()

def noise_test(train_loader, model, model_param, device):
  model.to(device)
  model.load_state_dict(torch.load(model_param))
  with torch.inference_mode():
    f = plt.figure(figsize=(10,6))
    f.add_subplot(3, 3, 1)
    t_img = train_loader[600].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 2)
    temp_img = train_loader[600] + 0.3*torch.randn_like(train_loader[i])
    f_img = temp_img.reshape(28, 28)
    plt.imshow(f_img, cmap='gray')
    f.add_subplot(3, 3, 3)
    g_img = model(temp_img.reshape(1,784)).reshape(28, 28)
    plt.imshow(g_img, cmap='gray')

    f.add_subplot(3, 3, 4)
    t_img = train_loader[1200].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 5)
    temp_img = train_loader[1200] + 0.3 * torch.randn_like(train_loader[i])
    f_img = temp_img.reshape(28, 28)
    plt.imshow(f_img, cmap='gray')
    f.add_subplot(3, 3, 6)
    g_img = model(temp_img.reshape(1, 784)).reshape(28, 28)
    plt.imshow(g_img, cmap='gray')

    f.add_subplot(3, 3, 7)
    t_img = train_loader[1800].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 8)
    temp_img = train_loader[1800] + 0.3*torch.randn_like(train_loader[i])
    f_img = temp_img.reshape(28, 28)
    plt.imshow(f_img, cmap='gray')
    f.add_subplot(3, 3, 9)
    g_img = model(temp_img.reshape(1,784)).reshape(28, 28)
    plt.imshow(g_img, cmap='gray')
    plt.show()

def interpolate_bottlenecks(model, image1, image2, n_steps):
  # Encode both input images to get the bottleneck tensors
  image1 = image1.reshape(1, 784)
  image2 = image2.reshape(1, 784)
  bottleneck1 = model.encode(image1)
  bottleneck2 = model.encode(image2)

  # Linearly interpolate between the two bottleneck tensors
  interpolated_bottlenecks = []
  for alpha in torch.linspace(0, 1, n_steps):
    interpolated_bottleneck = alpha * bottleneck1 + (1 - alpha) * bottleneck2
    interpolated_bottlenecks.append(interpolated_bottleneck)

  # Decode each interpolated bottleneck to get reconstructed images
  reconstructed_images = [model.decode(bottleneck) for bottleneck in interpolated_bottlenecks]
  print(len((reconstructed_images)))
  f = plt.figure(figsize=(10, 6))
  with torch.inference_mode():
    f.add_subplot(3, 3, 1)
    t_img = reconstructed_images[0].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 2)
    t_img = reconstructed_images[1].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 3)
    t_img = reconstructed_images[2].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 4)
    t_img = reconstructed_images[3].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    f.add_subplot(3, 3, 5)
    t_img = reconstructed_images[4].reshape(28, 28)
    plt.imshow(t_img, cmap='gray')
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-z", "--N_bottleneck", type=int, help="Specify the z (bottleneck) value")
  parser.add_argument("-e", "--epoch", type=int, help="Specify the number of epochs")
  parser.add_argument("-b", "--batch_size", type=int, help="Specify the batch size")
  parser.add_argument("-s", "--save_model", type=str, help="Specify the model save path")
  parser.add_argument("-p", "--plot_loss", type=str, help="Specify the loss plot path")

  args = parser.parse_args()



  transforms = torchvision.transforms
  MNIST = torchvision.datasets.MNIST
  train_transform = transforms.Compose([transforms.ToTensor()])
  train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
  data_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
  # test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=True)
  train_loader = []
  for i,batch in enumerate(data_loader):
    train_loader += batch[0]


  model = Model4layerMLP(N_Bottleneck = args.N_bottleneck)
  optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)
  loss_fn = nn.MSELoss()
  scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"



  # train(n_epochs = args.epoch, optimizer = optimizer, model = model, loss_fn = loss_fn, train_loader = train_loader[:2047],
  #     scheduler = scheduler, device = device, save_model_path = args.save_model, plot_loss_path = args.plot_loss)
  #
  #
  # test(train_loader,model,args.save_model,device)

  # noise_test(train_loader,model,args.save_model,device)

  # n_steps = 10
  # interpolate_bottlenecks(model, train_loader[600],train_loader[1200],n_steps)

    with (torch.inference_mode()):
      model.load_state_dict(torch.load('MLP.8.pth'))
      f = plt.figure(figsize=(14, 8))
      f.add_subplot(3, 11, 1)
      plt.imshow(train_loader[1].reshape(28,28), cmap = "gray")

      for i in range(1, 10):
        f.add_subplot(3, 11, i+1)
        img1 = model.encode(train_loader[1].reshape(1,784)) * (10-i) / 10
        img2 = model.encode(train_loader[2].reshape(1,784)) * i/10
        img = img1 + img2
        img = model.decode(img)
        img = img.reshape(28,28)
        plt.imshow(img, cmap="gray")

      f.add_subplot(3, 11, 11)
      plt.imshow(train_loader[2].reshape(28, 28), cmap="gray")
      plt.show()