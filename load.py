import numpy as np

def load_from_file(file_path):
    with open(file_path, 'r') as f:
        data = np.array([line.strip() for line in f.readlines()])
    return data

def load_data():
    train_sentences = load_from_file('data/train/sentences.txt')
    train_labels = load_from_file('data/train/labels.txt')

    val_sentences = load_from_file('data/val/sentences.txt')
    val_labels = load_from_file('data/val/labels.txt')

    test_sentences = load_from_file('data/test/sentences.txt')
    test_labels = load_from_file('data/test/labels.txt')

    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels