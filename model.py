import pickle
from load import load_data
from utils import *

def build_model(epochs = 2):
    train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels = load_data()

    ###################### SIDE WEIGHTS ######################

    sentence_vectorizer, vocab = get_sentence_vectorizer(train_sentences)

    tags = get_tags(train_labels)
    tag_map = make_tag_map(tags)

    side_weights = {'sentence_vectorizer': sentence_vectorizer,
                    'tag_map': tag_map,
                    'vocab': vocab}
    with open('model_weights/side_weights.pkl', 'wb') as file:
        pickle.dump(side_weights, file)
    print('sentence_vectorizer and tag_map saved')

    ###################### MODEL ######################

    train_dataset = generate_dataset(train_sentences,train_labels, sentence_vectorizer, tag_map)
    val_dataset = generate_dataset(val_sentences,val_labels,  sentence_vectorizer, tag_map)

    model = create_model(len(tag_map), len(vocab))
    model = compile_model(model)
    model = train_model(model, train_dataset, val_dataset, epochs)
    model.save_weights('model_weights/model_weights.weights.h5')
    print('Model saved')

    compute_accuracy(model, test_sentences, test_labels, sentence_vectorizer, tag_map)

build_model()