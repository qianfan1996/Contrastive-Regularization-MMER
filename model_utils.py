import torch
import numpy as np

from confusion_matrix import ConfusionMatrix

def run_epoch_train(model, iterator, optimizer, criterion, reg_ratio):
    model.train()

    epoch_loss = 0
    conf_mat = ConfusionMatrix(np.zeros((4, 4)))

    for batch, labels in iterator():
        optimizer.zero_grad()

        predictions = model(batch).squeeze(1)

        loss = criterion(predictions, labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + reg_ratio*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = epoch_loss / len(iterator)

    return average_loss, conf_mat

def run_epoch_eval(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    conf_mat = ConfusionMatrix(np.zeros((4, 4)))

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch)

            loss = criterion(predictions.float(), labels)
            epoch_loss += loss.item()
            conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = epoch_loss / len(iterator)

    return average_loss, conf_mat
