import datetime
import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from model_1 import MPL_3_layer_Classifier
from torch.utils.data import DataLoader
from torchsummary import summary
import argparse

def training(model,train_dataloader,loss_fn, scheduler, n_epochs, save_model_path):
    losses_train = []
    losses_test = []
    train_accuracy = []
    test_accuracy = []
    for epoch in range(1, n_epochs + 1):
        top1 = 0.0
        top1_train = 0.0
        print('epoch', epoch, "\t", "starts.....")
        for images, labels in train_dataloader:
            preds = model(images)
            loss_t = loss_fn(preds, labels)

            _, train_predictions = torch.max(preds, 1)
            train_accurate = torch.sum(train_predictions == labels).item()
            top1_train += train_accurate

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
        scheduler.step(loss_t)

        train_top1_accuracy = top1_train / len(training_data) * 100
        train_accuracy.append(train_top1_accuracy)

        losses_train += [loss_t.item()]

        for images, labels in test_dataloader:
            preds = model(images)
            loss_e = loss_fn(preds, labels)

            _, top1_predictions = torch.max(preds, 1)
            top1_accurate = torch.sum(top1_predictions == labels).item()
            top1 += top1_accurate

        top1_accuracy = top1 / len(test_data) * 100
        test_accuracy.append(top1_accuracy)
        losses_test += [loss_e.item()]

        print('{} Epoch {}, Training loss {}, Evaluation loss {}'.format(datetime.datetime.now()
                                                     , epoch, loss_t, loss_e))
    torch.save(model.state_dict(), save_model_path)
    return [losses_train,losses_test,train_accuracy,test_accuracy]

def evaluation(model, model_path, loss_fn, test_dataloader, losses_accuracies, figure_name):
    with (torch.inference_mode()):
        top1 = 0.0
        top3 = 0.0
        total_loss = 0.0
        model.load_state_dict(torch.load(model_path))
        for images, labels in test_dataloader:
            preds = model(images)
            _, top1_predictions = torch.max(preds, 1)
            top1_accurate = torch.sum(top1_predictions == labels).item()

            _, top3_predictions = torch.topk(preds, 3, dim=1)
            top3_accurate = top3_predictions == labels.view(-1, 1).expand_as(top3_predictions)
            top3_accurate = torch.sum(top3_accurate.any(dim=1)).item()

            top1 += top1_accurate
            top3 += top3_accurate

            loss = loss_fn(preds, labels)
            total_loss += loss.item()

        top1_accuracy = top1 / len(test_data) * 100
        top3_accuracy = top3 / len(test_data) * 100
        print('Final evaluation loss {}'.format(total_loss), f'Top1 Accuracy {top1_accuracy:.2f}%', f'Top3 Accuracy {top3_accuracy:.2f}%')



    plt.plot(losses_accuracies[0], label = "Train Loss")
    plt.plot(losses_accuracies[1], label = "Eval Loss")
    plt.title(f'Batch_{figure_name[0]} Nodes_{figure_name[1]} Dropout_{figure_name[2]} Regularization_{figure_name[3]} Optimizer_{figure_name[4]} LR_{figure_name[5]} Scheduler_{figure_name[6]}')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


    plt.plot(losses_accuracies[2], label = "Train Accuracy")
    plt.plot(losses_accuracies[3], label = "Eval Accuracy")
    plt.title("Accuracy")
    plt.ylabel('Percentage Accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(f'Batch_{figure_name[0]}Nodes_{figure_name[1]}Dropout_{figure_name[2]}Regularization_{figure_name[3]}Optimizer_{figure_name[4]}LR_{figure_name[5]}Scheduler_{figure_name[6]}.png')
    plt.show()








transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
training_data = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform= transforms
)

test_data = torchvision.datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform= transforms
)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--nodes", type=int, help="Specify the hidden layer nodes")
  parser.add_argument("-e", "--epoch", type=int, help="Specify the number of epochs")
  parser.add_argument("-b", "--batch_size", type=int, help="Specify the batch size")
  parser.add_argument("-lr", "--learning_rate", type=float, help="Specify the learning rate")
  parser.add_argument("-op", "--optimizer", type=str, help="Specify the Optimizer")
  parser.add_argument("-r", "--regularization", type=float, default=1e-15, help="Specify the weight decay value")
  parser.add_argument("-d", "--dropout", type=float, default=0.3, help="Specify the dropout rate")
  parser.add_argument("-s", "--scheduler", type=str, help="Specify the learning rate scheduler")
  parser.add_argument("-pth", "--path", type=str, default='trained_model.pth', help="Specify tge saved model path")

  args = parser.parse_args()

  train_dataloader = DataLoader(training_data, batch_size=args.batch_size)
  test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

  # print(len(training_data))
  # print(len(test_data))
  # print(len(test_dataloader))
  # print(training_data[0][0].shape)
  # print(training_data[0][0].squeeze().shape)
  # plt.imshow(training_data[0][0].squeeze(), cmap="gray")
  # plt.show()
  model = MPL_3_layer_Classifier(N_tanh_activate=args.nodes, p=args.dropout)
  lr = args.learning_rate
  epochs = args.epoch
  optimizer = str(args.optimizer).upper()
  weight_decay = float(args.regularization)
  loss_function = nn.CrossEntropyLoss()
  scheduler = str(args.scheduler).upper()
  save_model_path = args.path

  if optimizer.startswith("ADAM") or optimizer == "1":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opti_name = "ADAM"
  elif optimizer.startswith("SGD") or optimizer == "2":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,  weight_decay=weight_decay)
    opti_name = "SGD"
  elif optimizer.startswith("RMS") or optimizer == "3":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, weight_decay=weight_decay)
    opti_name = "RMSprop"

  if scheduler.startswith("ROP") or scheduler == "1":
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
      sched_name = "ReduceOnPlateau"
  elif scheduler.startswith("STEP") or scheduler == "2":
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
      sched_name = "StepLR"
  elif scheduler.startswith("ONEC") or scheduler == "3":
      scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs)
      sched_name = "OneCycle"

  figure_name = [str(args.batch_size), str(args.nodes), str(args.dropout), 'L2_'+str(args.regularization), opti_name, str(lr), sched_name]

  # print(optimizer)
  # print(scheduler)
  # data1 = test_data[0][0]
  # plt.imshow(data1.squeeze(), cmap="gray")
  # plt.show()
  # data1 = torch.reshape(data1,(1,784))
  # print(data1.shape)
  # data_1_tra = model.forward(data1)
  # print(data_1_tra)
  # for images, labels in train_dataloader:
  #     plt.imshow(images[1].squeeze(), cmap="gray")
  #     plt.show()
  #     print(labels[1])
  train_losses = training(model, train_dataloader, loss_function, scheduler, epochs, save_model_path)
  evaluation(model, save_model_path, loss_function, test_dataloader, train_losses, figure_name)




