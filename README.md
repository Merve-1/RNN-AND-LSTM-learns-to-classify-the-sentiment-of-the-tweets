# Sentiment Analysis using RNN and LSTM

## Overview
This repository contains code for sentiment analysis using Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) models. The models are implemented using TensorFlow in a Google Colab environment. The dataset used for training and evaluation is a collection of tweets labeled with sentiment values (positive or negative).

## Configuration
To run the code, make sure you have the necessary libraries installed. The following dependencies are required:
```python
from google.colab import drive
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tabulate import tabulate
```

## Drive Connection
Mount your Google Drive to access the dataset:
```python
drive.mount('/content/drive')
```

## Load Dataset
Load the dataset from your Google Drive. The dataset should be in CSV format with columns: 'target', 'id', 'date', 'query-flag', 'user', 'tweet_text'. Convert sentiment values to 0 for negative and 1 for positive:
```python
df = pd.read_csv('/content/drive/My Drive/training.csv', encoding='latin-1')
df.columns = ['target', 'id', 'date', 'query-flag', 'user', 'tweet_text']
df['target'] = np.where(df['target'] == 4, 1, 0)
```

## Dataset Splitting
Split the dataset into training (80%), validation (10%), and testing (10%) sets:
```python
train_df, test_validation_df = train_test_split(df, test_size=0.2, random_state=123)
validation_df, test_df = train_test_split(test_validation_df, test_size=0.5, random_state=123)
```

Create TensorFlow datasets from the training and testing DataFrames:
```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['tweet_text'].values, train_df['target'].values))
test_dataset = tf.data.Dataset.from_tensor_slices((test_df['tweet_text'].values, test_df['target'].values))

# Apply batch and prefetch functions to the datasets
BATCH_SIZE = 64
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
```

## Model Architecture
### RNN Model
```python
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

### LSTM Model
```python
model2 = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

## Model Training
Compile and train both models:
```python
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=5, validation_data=test_dataset, validation_steps=30)
```

## Model Evaluation
Evaluate the models on the test dataset:
```python
test_loss1, test_acc1 = model.evaluate(test_dataset)
test_loss2, test_acc2 = model2.evaluate(test_dataset)

# Display test results in a tabular format
table = [
    ["RNN Model", test_loss1, test_acc1],
    ["LSTM Model", test_loss2, test_acc2]
]
headers = ["Model", "Test Loss", "Test Accuracy"]
print(tabulate(table, headers=headers))
```
