import wfdb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import backend as K
from tensorflow.keras.models import load_model


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
model = load_model('classification_model_capsule_v3_with_smote_and_augmentation.keras', compile=False, custom_objects={'squash': squash}, safe_mode=False)
predictions = model.predict(X_test)

threshold = 0.5  # Adjust this threshold based on your model's output
binary_predictions = (predictions > threshold).astype(np.uint8)
y_true_flat = y_test.flatten()
y_pred_flat = binary_predictions.flatten()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

confusion_mat = confusion_matrix(y_true_flat, y_pred_flat)

# Plotting the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_true_flat, y_pred_flat))
