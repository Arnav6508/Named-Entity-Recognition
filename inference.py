import pickle
import numpy as np
import tensorflow as tf
from load import load_weights
from utils import compute_accuracy

def predict(sentence):

    tag_map, sentence_vectorizer, model = load_weights()

    vectorized_sentence = sentence_vectorizer(sentence)
    vectorized_sentence = tf.expand_dims(vectorized_sentence, axis = 0) ## add batch dim

    outputs = model.predict(vectorized_sentence)
    outputs = np.argmax(outputs, axis = -1)
    outputs = outputs[0]

    labels = list(tag_map.keys()) 
    pred = [] 
    for tag_idx in outputs:
        pred_label = labels[tag_idx]
        pred.append(pred_label)
    
    for word, ner_label in zip(sentence.split(' '), pred):
        print(word,':',ner_label)

    return pred

def test_accuracy(test_sentences, test_labels):
    tag_map, sentence_vectorizer, model = load_weights()
    acc = compute_accuracy(model, test_sentences, test_labels, sentence_vectorizer, tag_map)
    return acc
