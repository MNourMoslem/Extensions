import torch
from torch import nn, optim
import torchvision

import matplotlib.pyplot as plt

import os
import zipfile

import pathlib

import requests

import random

def download_zipfile(link:str, target_folder:str, file_name:str, keep_zipfile:bool = True,):
  '''
  This function provided to help you upload zipfiles from the internet (Githup)
  to your local folder.

  link: The link of the file that wanted to be upload
  target_folder: The flie where the file will be unzipped in
  file_name: The name of the uploaded file (it should be .zip file)
  keep_zip_file: Will keep the zipfile if "True",  or delete it if "Flase"
  '''

  data_dir = pathlib.Path(f'{target_folder}/')
  data_zipfile_dir = data_dir / file_name

  if data_zipfile_dir.is_dir():
    print('Data zip file is Already Exist')
  else:
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_zipfile_dir, 'wb') as f:
      request = requests.get(link)
      f.write(request.content)

    with zipfile.ZipFile(data_zipfile_dir, 'r') as zip_dir:
      zip_dir.extractall(data_dir)

  if not keep_zipfile:
    os.remove(data_zipfile_dir)

def create_ImgDataloader(train_dir:str,
                        test_dir:str,
                        train_transform: torchvision.transforms,
                        test_transform: torchvision.transforms,
                        batch_size:int,
                        test_batch_size:int=None):
  """
  This function takes the train/test diractories and convert them into
  torchvisoin.datasets.ImageFolder --> torch.utils.data.Dataloader and will
  return them on the shape:
  Tuple[train_dataloader, test_dataloader, class_names]

  This function has been made for the image dataset, it won't work if its
  another datatype

  train_dir: The diractory of the train set (example: data/.../train)
  test_dir: The diractory of the test set (example: data/.../test)
  train_transform: The transform the will be applied to the train data
  test_transform: The transform the will be applied to the test data
  batch_size: The number of batches you want to split your data according to
  test_batch_size: if True it won't apply a batch to the test data
  """

  test_batch_size = test_batch_size if test_batch_size != None else batch_size

  train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
  test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

  train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)

  return train_dataloader, test_dataloader, train_data.classes

def train_model(model:nn.Module,
                train_dataloader:torch.utils.data.DataLoader,
                test_dataloader:torch.utils.data.DataLoader,
                optimizer: torch.optim,
                loss_fn:torch.nn,
                epochs:int=1,
                train_epochs:int=1,
                device:torch.device='cpu'):
  """
  This Function train your data and then test it module according to the hyperparameters bellow.
  Train Function will also returns -> Tuple[train_loss, test_loss, train_accuracy, test_accuracy]

  train/test losses will be the avg. of the losses / number of batch size

  Hyperparameter:
  model: The model is wanted to be trained
  train_dataloader: The dataloader of the train data, it has to by of the type of torch.uilts.data.DataLoader
  test_dataloader: The dataloader of the test data, it has to by of the type of torch.uilts.data.DataLoader
  optimizer: The optimizer function the optimizer the parameters of the model
  loss_fn: The loss function that compute the loss
  epochs: The number of epochs to train and test the data (defult: 1)
  train_epochs: The number of epochs to train before the testing (defult: 1)
  device: The device wanted to compute the functionality of the process (defult: cpu)
  """

  def train_step(dataloader, model, optimizer, loss_fn, epochs=1, device=device):
    model.train()
    model.to(device)
    correct, train_loss = 0, 0

    for i in range(epochs):
      for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_logits = model(x)

        loss = loss_fn(y_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i+1 == epochs:
          correct += torch.eq(y_logits.argmax(1), y).float().sum().item()
          train_loss += loss.item()
    acc = correct/len(dataloader.dataset)*100
    train_loss /= len(dataloader)

    print(f'Train Loss: {train_loss:.7f} | Train Accuracy: {acc:.2f}%')
    return train_loss, acc

  def test_step(dataloader, model, loss_fn, device=device):
    model.eval()
    model.to(device)
    correct, test_loss = 0, 0

    with torch.inference_mode():
      for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_logits = model(x)
        loss = loss_fn(y_logits, y)
        correct += torch.eq(y_logits.argmax(1), y).float().sum().item()
        test_loss += loss.item()

    acc = correct/len(dataloader.dataset)*100
    test_loss /= len(dataloader)

    print(f'Test Loss: {test_loss:.5f} | Test Accuracy: {acc:.2f}%')
    return test_loss, acc

  train_loss_list = []
  test_loss_list = []
  train_acc_list = []
  test_acc_list = []

  for i in range(epochs):
    print(f'\nEpoch {i+1}:\n------------------------------')
    train_loss, train_acc = train_step(train_dataloader, model, optimizer, loss_fn, train_epochs)
    test_loss, test_acc = test_step(test_dataloader, model, loss_fn)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
  print('Done')

  return train_loss_list, test_loss_list, train_acc_list, test_acc_list

def save_model(model:nn.Module, name:str, diractory:str = None):
  """
  This Fucntion helps you to save your model to the diractory chossen

  hyperparameters:
  model: The model wanted to be save its wights
  name: The name of the wights file (it shoudl be an .pth type file)
  diractory: The diractory address to where to save the model (for ex. xxx/.../yyy)
  """
  dir = f'{diractory}/{name}' if diractory != None else name

  torch.save(model.state_dict(), dir)
  print('Weights has been saved succesfully!')

def load_model(model:str, diractory:str):
  """
  This Fucntion helps you to load the weights to the model chossen from  the diractory

  hyperparameters:
  model: The model wanted to load the weights to
  diractory: The diractory address of the saved weights (for ex. xxx/.../model_weights.pth)
  """

  model.load_state_dict(torch.load(diractory))
  print('Weights has been Loaded succesfully!')

def predect_and_plotImages(model: nn.Module,
                           dataloader:torch.utils.data.DataLoader,
                           n_predections:int,
                           class_names:list, 
                           seed:int=None
                           ):
  """
  This Functoin helps to predict certain amount of predections and compare them with the labels.
  The comparing will be by showing tge label and the predections as a title to the image input.
  The input being choosed randomly or depending on the seed.
  Lastly the maximum number of predections/images is 5.

  Hyperparameters:
  model: The model that will do the predictions
  dataloader: The dataloader of the data will be used in the function (must be in  type torch.utils.data.DataLoader)
  n_predections: The number of predections wanted to be shown, maximum 5
  class_namse: All class names of the predections
  seed: The random seed of the randomness function
  """

  row, col = 1  , 5 if n_predections > 5 else n_predections
  if seed != None:
    random.seed(seed)

  if n_predections>5:
    print("Can not display more than 5 images in a row")

  fig = plt.figure(figsize=(16, 4))
  for i in range(1, row*col+1):
    fig.add_subplot(row, col, i)
    rand_n = random.randint(0, len(dataloader.dataset)-1)
    img, label = dataloader.dataset[rand_n]
    label_pred = model(img.unsqueeze(0)).argmax(1).item()
    plt.imshow(img.permute(1, 2, 0))
    if label_pred == label:
      plt.title(f'Pred: {class_names[label_pred]} | Label: {class_names[label]}', c='green')
    else:
      plt.title(f'Pred: {class_names[label_pred]} | Label: {class_names[label]}', c='red')
    plt.axis(False)
  plt.show()

def set_seed(seed:int=None):
  """
  Sets the manule seeds to each cpu and cuda in torch
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
