import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(text, label):
    return text.numpy().decode('utf-8'), label.numpy()

def load_and_prepare_data(max_length=200, num_words=10000):
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']

    # Convert to numpy arrays
    train_examples = list(map(preprocess_data, *tfds.as_numpy(train_data)))
    test_examples = list(map(preprocess_data, *tfds.as_numpy(test_data)))

    # Tokenize the text
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts([x[0] for x in train_examples])

    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences([x[0] for x in train_examples])
    test_sequences = tokenizer.texts_to_sequences([x[0] for x in test_examples])

    # Pad sequences
    X_train = pad_sequences(train_sequences, maxlen=max_length)
    X_test = pad_sequences(test_sequences, maxlen=max_length)

    # Labels
    y_train = [x[1] for x in train_examples]
    y_test = [x[1] for x in test_examples]

    return {'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}