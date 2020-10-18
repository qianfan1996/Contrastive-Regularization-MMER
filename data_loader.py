from os.path import isfile
import numpy as np

from utils import timeit
from batch_iterator import BatchIterator

LINGUISTIC_DATASET_PATH = "data/linguistic_glove_features.npy"
LINGUISTIC_LABELS_PATH = "data/linguistic_glove_labels.npy"
LINGUISTIC_DATASET_ASR_PATH = "data/linguistic_glove_features_asr.npy"
LINGUISTIC_LABELS_ASR_PATH = "data/linguistic_glove_labels_asr.npy"
SPECTROGRAMS_FEATURES_PATH = "data/spectrograms_features.npy"
SPECTROGRAMS_LABELS_PATH = "data/spectrograms_labels.npy"

def split_dataset_skip(dataset_features, dataset_labels, split_ratio=0.2):
    """Splittng dataset into train/val sets by taking every nth sample to val set"""
    skip_ratio = int(1/split_ratio)
    all_indexes = list(range(dataset_features.shape[0]))
    test_indexes = list(range(0, dataset_features.shape[0], skip_ratio))
    train_indexes = list(set(all_indexes) - set(test_indexes))
    val_indexes = train_indexes[::10]
    train_indexes = list(set(train_indexes) - set(val_indexes))

    test_features = dataset_features[test_indexes]
    test_labels = dataset_labels[test_indexes]
    val_features = dataset_features[val_indexes]
    val_labels = dataset_labels[val_indexes]
    train_features = dataset_features[train_indexes]
    train_labels = dataset_labels[train_indexes]

    assert test_features.shape[0] == test_labels.shape[0]
    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]

    return test_features, test_labels, val_features, val_labels, train_features, train_labels

def create_batches(test_features, test_labels, val_features, val_labels, train_features, train_labels, batch_size):
    test_iterator = BatchIterator(test_features, test_labels)
    train_iterator = BatchIterator(train_features, train_labels, batch_size)
    validation_iterator = BatchIterator(val_features, val_labels)
    return test_iterator, train_iterator, validation_iterator

def load_dataset(features_path, labels_path):
    """Extracting & Saving dataset"""
    if not isfile(features_path) or not isfile(labels_path):
        print("Dataset not found.")

    # Loading dataset
    dataset = np.load(features_path)
    labels = np.load(labels_path)
    print("Dataset loaded.")

    assert dataset.shape[0] == labels.shape[0]

    return split_dataset_skip(dataset, labels)

@timeit
def load_spectrogram_dataset():
    return load_dataset(SPECTROGRAMS_FEATURES_PATH, SPECTROGRAMS_LABELS_PATH)

@timeit
def load_linguistic_dataset(asr=False):
    dataset_path = LINGUISTIC_DATASET_ASR_PATH if asr else LINGUISTIC_DATASET_PATH
    labels_path = LINGUISTIC_LABELS_ASR_PATH if asr else LINGUISTIC_LABELS_PATH
    return load_dataset(dataset_path, labels_path)

