import pickle
import numpy as np
import tensorflow as tf
from utils import create_model

def predict(sentence):

    with open('model_weights/side_weights.pkl', 'rb') as file:
        side_weights = pickle.load(file)
    tag_map = side_weights['tag_map']
    sentence_vectorizer = side_weights['sentence_vectorizer']
    vocab = side_weights['vocab']

    ## B4 loading weights model needs to be built so it knows input dim
    model = create_model(len(tag_map), len(vocab), embedding_dim = 50)
    model.build(input_shape=(None, len(vocab)+1)) 
    model.load_weights('model_weights/model_weights.weights.h5')

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
