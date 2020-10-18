import os
from time import localtime, strftime
import json

import torch

from model_utils import run_epoch_eval, run_epoch_train
from utils import get_datetime, log_major, log_success, get_device


MODEL_PATH = "saved_models"


def train(model, cfg, test_iterator, train_iterator, validation_iterator):
    tmp_run_path = "tmp/" + cfg.model_name + "_" + get_datetime()
    model_weights_path = "{}/{}".format(tmp_run_path, cfg.model_weights_name)
    model_config_path = "{}/{}".format(tmp_run_path, cfg.model_config_name)
    result_path = "{}/result.txt".format(tmp_run_path)
    os.makedirs(tmp_run_path, exist_ok=True)
    json.dump(cfg.to_json(), open(model_config_path, "w"))

    # Defining loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(get_device())

    train_loss = 999
    best_val_loss = 999
    train_acc = 0
    epochs_without_improvement = 0

    # Running training
    for epoch in range(cfg.n_epochs):
        train_iterator.shuffle()
        if epochs_without_improvement == cfg.patience:
            break

        val_loss, val_cm = run_epoch_eval(model, validation_iterator, criterion)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), model_weights_path)
            best_val_loss = val_loss
            best_val_acc = val_cm.accuracy
            best_val_unweighted_acc = val_cm.unweighted_accuracy
            epochs_without_improvement = 0
            log_success(" Epoch: {} | Validation loss improved to {:.4f} | Validation accuracy: {:.4f} | Weighted validation accuracy: {:.4f} | train loss: {:.4f} | train accuracy: {:.4f} | saved model to {}.".format(
                epoch, best_val_loss, best_val_acc, best_val_unweighted_acc, train_loss, train_acc, model_weights_path))

        train_loss, train_cm = run_epoch_train(model, train_iterator, optimizer, criterion, cfg.reg_ratio)
        train_acc = train_cm.accuracy

        epochs_without_improvement += 1

    model.load_state_dict(torch.load(model_weights_path))
    test_loss, test_cm = run_epoch_eval(model, test_iterator, criterion)

    result = f'| Epoch: {epoch+1} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_cm.accuracy*100:.2f}% | Weighted Test Accuracy: {test_cm.unweighted_accuracy*100:.2f}%\n Confusion matrix:\n {test_cm}'
    log_major("Train accuracy: {}".format(train_acc))
    log_major(result)
    log_major("Hyperparameters:{}".format(cfg.to_json()))
    with open(result_path, "w") as file:
        file.write(result)

    output_path = "{}/{}_{:.4f}Acc_{:.4f}UAcc_{}".format(MODEL_PATH, cfg.model_name, test_cm.accuracy, test_cm.unweighted_accuracy, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    os.renames(tmp_run_path, output_path)
