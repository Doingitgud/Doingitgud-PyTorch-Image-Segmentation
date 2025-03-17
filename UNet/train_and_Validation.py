from tqdm import tqdm
from configuration import DEVICE
import torch


def train_fn(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(data_loader)




def eval_fn(data_loader, model):

   model.eval()
   total_loss = 0.0

   with torch.no_grad():
     for images, masks in tqdm(data_loader):

       images = images.to(DEVICE)
       masks = masks.to(DEVICE)

       logits, loss = model(images, masks)

       total_loss += loss.item()

     return total_loss / len(data_loader)