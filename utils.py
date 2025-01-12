import tensorflow as tf
import matplotlib.pyplot as plt

def get_sentence_vectorizer(sentences):
    sentence_vectorizer = tf.keras.layers.TextVectorization(standardize = None)
    sentence_vectorizer.adapt(sentences)
    vocab = sentence_vectorizer.get_vocabulary()
    return sentence_vectorizer, vocab

def get_tags(labels):
    tag_set = set()
    for label in labels: 
        for tag in label.split(' '): 
            tag_set.add(tag)
    tags = list(tag_set)
    tags.sort()
    return tags

def make_tag_map(tags):
    tag_map = {}
    for i, tag in enumerate(tags): tag_map[tag] = i
    return tag_map

def label_vectorizer(labels, tag_map):
    vectorized_labels = []
    for label in labels:
        curr_vec_label = []
        for tag in label.split(' '): curr_vec_label.append(tag_map[tag])
        vectorized_labels.append(curr_vec_label)

    vectorized_labels = tf.keras.utils.pad_sequences(vectorized_labels, padding = 'post', value = -1)
    return vectorized_labels

def generate_dataset(sentences, labels, sentence_vectorizer, tag_map):
    sentences_ids = sentence_vectorizer(sentences)
    labels_ids = label_vectorizer(labels, tag_map)
    dataset = tf.data.Dataset.from_tensor_slices((sentences_ids, labels_ids))
    return dataset

def masked_loss(y_true, y_pred):
    # Since we used log softmax, the pred is not between finite range and hence not normalized
    # Due to these un-normalized pred, we use from_logits = True
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, ignore_class = -1)
    loss = loss_fn(y_true, y_pred)
    return loss

def masked_accuracy(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred, axis = -1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    mask = y_true>=0
    mask = tf.cast(mask, tf.float32)

    y_correct = tf.equal(y_true, y_pred)
    y_correct = tf.cast(y_correct, tf.float32)
    y_correct_with_mask = y_correct*mask

    return tf.reduce_sum(y_correct_with_mask)/tf.reduce_sum(mask)

def create_model(len_tags, vocab_size, embedding_dim = 50, dropout_rate = 0.2):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size+1, embedding_dim, mask_zero = True))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = embedding_dim, return_sequences= True)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units = len_tags, activation = tf.nn.log_softmax))

    return model

def compile_model(model):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.01),
        loss = masked_loss,
        metrics = [masked_accuracy]
    )
    return model

def make_plot(history, metric):
    plt.plot(history.history[f'{metric}'], label = f'{metric}')
    plt.plot(history.history[f'val_{metric}'], label = f'Val_{metric}')
    plt.title(f'{metric} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

def train_model(model, train_dataset, val_dataset, epochs = 2, BATCH_SIZE = 64):
    history = model.fit(train_dataset.batch(BATCH_SIZE),
              validation_data = val_dataset.batch(BATCH_SIZE),
              shuffle = True,
              epochs = epochs
              )
    
    make_plot(history, 'loss')
    make_plot(history, 'masked_accuracy')

    return model

def compute_accuracy(model, test_sentences, test_labels, sentence_vectorizer, tag_map):
    test_ids = sentence_vectorizer(test_sentences)
    label_ids = label_vectorizer(test_labels, tag_map)

    y_true = label_ids
    y_pred = model.predict(test_ids)
    acc = masked_accuracy(y_true, y_pred)

    print('Accuracy of model on test set is:', acc)
    return acc