import pickle
import numpy as np
from utils import create_model

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

def load_weights():
    with open('model_weights/side_weights.pkl', 'rb') as file:
        side_weights = pickle.load(file)
    tag_map = side_weights['tag_map']
    sentence_vectorizer = side_weights['sentence_vectorizer']

    vocab = sentence_vectorizer.get_vocabulary()

    ## B4 loading weights model needs to be built so it knows input dim
    model = create_model(len(tag_map), len(vocab), embedding_dim = 50)
    model.build(input_shape=(None, len(vocab)+1)) 
    model.load_weights('model_weights/model_weights.weights.h5')

    return tag_map, sentence_vectorizer, model