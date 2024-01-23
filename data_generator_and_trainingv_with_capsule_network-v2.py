import wfdb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import layers, models
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import backend as K

def squash(x, axis=-1):
    squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / K.sqrt(squared_norm + K.epsilon())

def getLabel(value):
    if value == 'N':
        return 0
    else:
        return 1


myDataset = []
heartbeats = []
labels = []

data_path = '/Users/shamistanhuseynov/PycharmProjects/pythonProject/mit-bih-arrhythmia-database-1.0.0'

record_list = [f.replace('.hea', '') for f in os.listdir(data_path) if f.endswith('.hea')]

df = pd.DataFrame(columns=['Heartbeat', 'Annotation'])

for record_name in record_list:
    record_path = os.path.join(data_path, record_name)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    window_size = 300

    for i in range(0, len(annotation.symbol)):
        if annotation.symbol[i] in ['N', 'V', 'F']:
            center = annotation.sample[i]
            window_start = max(0, center - window_size // 2)
            window_end = min(len(record.p_signal), center + window_size // 2)
            heartbeat = tuple(record.p_signal[window_start:window_end, 0])

            if len(heartbeat) == window_size:
                heartbeats.append(heartbeat)
                labels.append(getLabel(annotation.symbol[i]))

data = np.array(heartbeats)
data = data.reshape((data.shape[0], data.shape[1]))
labels = np.array(labels)

smote = SMOTE(random_state=28)
X_resampled, y_resampled = smote.fit_resample(data, labels)

# One-hot encode labels
labels_one_hot = tf.keras.utils.to_categorical(y_resampled, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, labels_one_hot, test_size=0.2, random_state=42)

# datagen = TimeseriesGenerator(X_train, y_train, length=window_size, batch_size=16, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=True)

def CapsuleNetwork(input_shape):
    inputs = layers.Input(shape=input_shape)
    inputs = layers.BatchNormalization()(inputs)

    conv1 = layers.Conv1D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(conv1)
    conv3 = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(conv2)
    conv4 = layers.Conv1D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(conv3)
    primary_caps = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(conv4)

    digit_caps = layers.Conv1D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(primary_caps)
    digit_caps = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(digit_caps)

    skip_connection = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(primary_caps)
    digit_caps = layers.Add()([skip_connection, digit_caps])

    digit_caps = layers.Lambda(squash)(digit_caps)

    forecast = layers.Conv1D(filters=2, kernel_size=1, strides=1, padding='same', activation='sigmoid')(digit_caps)
    forecast = layers.Flatten()(forecast)
    forecast = layers.Dense(2)(forecast)

    model = models.Model(inputs=inputs, outputs=forecast)
    return model


# Create and compile the model
model = CapsuleNetwork((300, 1))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with data augmentation using the generator
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Save the model
model.save("classification_model_capsule_v3_with_smote_and_augmentation.keras")

# Evaluate the model
accuracy = model.evaluate(X_test, y_test, batch_size=16)[1]
print("Accuracy:", accuracy)
