"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from dvclive import Live

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               batch_tfms,
               loss_fn: torch.nn.Module, 
               acc_fns: List,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  model.train()
  accum_loss = 0
  for batch in dataloader:

      if batch_tfms is not None:
          batch = batch_tfms(batch)

      X = batch['image'].to(device)
      y = batch['mask'].type(torch.long).to(device)
      pred = model(X)
      loss = loss_fn(pred, y)

      # BackProp
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # update the accum loss
      accum_loss += float(loss) / len(dataloader)

      # reset the accuracies metrics
      acc = [0.] * len(acc_fns)

      for i, acc_fn in enumerate(acc_fns):
                  acc[i] = float(acc[i] + acc_fn(pred, y)/len(dataloader))
  return accum_loss, acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              batch_tfms,
              loss_fn: torch.nn.Module,
              acc_fns: list,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    acc_fns: A list of accuracy functions
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  model.eval()
  accum_val_loss = 0
  # Testing against the validation dataset
  if acc_fns is not None and dataloader is not None:
      # reset the accuracies metrics
      acc = [0.] * len(acc_fns)

      with torch.no_grad():
          for batch in dataloader:

              if batch_tfms is not None:
                  batch = batch_tfms(batch)                    

              X = batch['image'].type(torch.float32).to(device)
              y = batch['mask'].type(torch.long).to(device)

              pred = model(X)
              val_loss = loss_fn(pred, y)
              accum_val_loss += float(val_loss)/len(dataloader)

              for i, acc_fn in enumerate(acc_fns):
                  acc[i] = float(acc[i] + acc_fn(pred, y)/len(dataloader))

  return accum_val_loss, acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          batch_tfms,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          acc_fns,
          epochs: int,
          callbacks: Dict,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  best_chkpt_score = float("-inf")
  for epoch in tqdm(range(1, epochs+1)):
      train_loss, train_acc = train_step(model, train_dataloader, batch_tfms, loss_fn, acc_fns, optimizer, device)
      test_loss, test_acc = test_step(model, test_dataloader, batch_tfms, loss_fn, acc_fns, device)
      # at the end of the epoch, print the errors, etc.
      print(f'Epoch {epoch}: Train Loss={train_loss:.5f} | Train Acc={[round(a, 3) for a in train_acc]}')
      print(f'Test Loss={test_loss:.5f} | Test Acc={[round(a, 3) for a in test_acc]}')
      # Check if validation loss improved
      chkpt_score = test_acc[callbacks.get('metric_index')]
      if  chkpt_score > best_chkpt_score:
          best_chkpt_score = chkpt_score
          torch.save(model.state_dict(), callbacks.get("save_model_path"))
          print(f'Saving model with test score: {best_chkpt_score:.4f} at epoch {epoch}')

      train_params = {'epochs': epochs}
      # Track metrics
      with Live(save_dvc_exp=True) as live:
          for param_name, param_value in train_params.items():
              live.log_param(param_name, param_value)
          live.log_metric("model_saving_score", chkpt_score)
          live.log_metric("Train loss", train_loss)
          live.log_metric("Val_loss", test_loss)