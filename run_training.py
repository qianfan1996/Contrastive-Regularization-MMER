import argparse

from models import AttentionLSTM as RNN, CNN
from batch_iterator import BatchIterator
from data_loader import load_linguistic_dataset, load_spectrogram_dataset
from utils import set_default_tensor
from config import LinguisticConfig, AcousticSpectrogramConfig
from train import train

num_iteration = 20

if __name__ == "__main__":
    for i in range(num_iteration):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model_type", type=str, default="linguistic")
        args = parser.parse_args()
        set_default_tensor()

        if args.model_type == "linguistic":
            cfg = LinguisticConfig()
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
            model = RNN(cfg)
        elif args.model_type == "acoustic-spectrogram":
            cfg = AcousticSpectrogramConfig()
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_spectrogram_dataset()
            model = CNN(cfg)
        else:
            raise Exception("model_type parameter has to be one of [linguistic|acoustic-spectrogram]")

        """Creating data generators"""
        test_iterator = BatchIterator(test_features, test_labels)
        train_iterator = BatchIterator(train_features, train_labels, cfg.batch_size)
        validation_iterator = BatchIterator(val_features, val_labels)

        """Running training"""
        train(model, cfg, test_iterator, train_iterator, validation_iterator)
