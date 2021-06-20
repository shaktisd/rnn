# Imports
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from tensorflow.keras import layers, models, optimizers
import pickle

# Constants
TRAINING_DATA_PATH = 'FinancialPhraseBank-v1.0/Sentences_50Agree.txt'
VECTOR_DIM = 768


# API
def get_data(path=TRAINING_DATA_PATH):
    df = pd.read_csv(path, sep='@', names=['sentence', 'label'])
    return df


def get_model(model_name='paraphrase-distilroberta-base-v1'):
    model = SentenceTransformer(model_name)
    return model


def create_nn(number_of_classes):
    input_size = VECTOR_DIM
    # create input layer
    input_layer = layers.Input((input_size,))

    # create hidden layer
    hidden_layer = layers.Dense(50, activation="relu")(input_layer)

    dropoutput_layer = layers.Dropout(0.25)(hidden_layer)

    # create output layer
    output_layer = layers.Dense(number_of_classes, activation="softmax")(dropoutput_layer)

    c = models.Model(inputs=input_layer, outputs=output_layer)
    return c


# Main
if __name__ == '__main__':
    embedding_model = get_model()
    df = get_data()

    number_of_classes = df['label'].nunique()
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])

    embeddings = embedding_model.encode(df['sentence'], show_progress_bar=True)
    df['embeddings'] = embeddings.tolist()
    print('df.shape', df.shape)

    train_df, temp_df = train_test_split(df[['embeddings', 'encoded_label']], random_state=456, test_size=0.3)
    val_df, test_df = train_test_split(temp_df, random_state=789, test_size=0.5)

    x_train = np.array(train_df['embeddings'].tolist())
    x_test = np.array(test_df['embeddings'].tolist())
    x_val = np.array(val_df['embeddings'].tolist())
    y_train = np.array(train_df['encoded_label'].tolist())
    y_train = y_train.reshape(-1, 1)
    y_test = np.array(test_df['encoded_label'].tolist())
    y_test = y_test.reshape(-1, 1)
    y_val = np.array(val_df['encoded_label'].tolist())
    y_val = y_val.reshape(-1, 1)

    print('Training Data', x_train.shape, y_train.shape)
    print('Validation Data', x_val.shape, y_val.shape)
    print('Test Data', x_test.shape, y_test.shape)

    classifier = create_nn(number_of_classes)

    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy')

    classifier.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

    # # predict the labels on validation dataset
    predictions = classifier.predict(x_test)

    predictions = predictions.argmax(axis=-1)

    print('Accuracy ', metrics.accuracy_score(predictions, y_test))
    # model's performance
    print(metrics.classification_report(y_test, predictions))

    print('Saving Model')
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    classifier.save('classifier_model.bin')
