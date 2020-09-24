import snowflake.connector
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import os
import numpy as np
import datetime
import json

with open('configs.json') as f:
  configs = json.load(f)

user        = configs['SF_USER']
password    = configs['SF_PASSWORD']
account     = configs['ACCOUNT']
warehouse   = 'COMPUTE_WH'
database    = 'TRADING_DB'
schema      = 'PUBLIC'

conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    warehouse=warehouse,
    database=database,
    schema=schema
)

conn.cursor().execute('USE warehouse COMPUTE_WH')
conn.cursor().execute('USE TRADING_DB.BTC_USD')

table = 'TRADE_PROFIT_SIGNALS'

query = f'''
select * from TRADING_DB.BTC_USD.{table};
'''
cur = conn.cursor().execute(query)
df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
print(f'There are {len(df)} records read from Snowflake table {table}')

df['buy_binary'] = np.where(df.ORDER_SIGNAL=='buy', 1, 0)

df['TIME'].min()
df['TIME'].max()
df_subset = df[df.TIME >= '2020-09-22']
print(f"There are {len(df_subset)} records in the subset with min date {df_subset['TIME'].min()}")

print("Class Counts of Training Subset:")
df_subset.groupby('buy_binary')['SEQUENCE'].count()


# Create weights to combat class imbalance
def create_class_imb_weights(df, num_outcomes, response):

    no_pos_count, pos_count = np.bincount(df[response])
    total_count = len(df[response])

    weight_neg = (1 / no_pos_count) * (total_count) / num_outcomes
    weight_pos = (1 / pos_count) * (total_count) / num_outcomes

    #TODO: enumerate for >2 outcomes
    class_weights = {0: weight_neg, 1: weight_pos}
    print("Class Weights: ", class_weights)

class_weights = create_class_imb_weights(df_subset, 2, 'buy_binary')

# Choose predictor fields
fields = list(df_subset.columns.values)
print("All Fields:")
print(fields)

fields_to_remove = ('buy_binary','PROFIT_SIGNAL','SEQUENCE',
                    'ORDER_SIGNAL','TIME','PREV_MINUTE')

fields_pred = [e for e in fields if e not in fields_to_remove]

print('Response and Fields to be used as predictors:')
print(fields_pred)


# training and test sets
X = df_subset[fields_pred]
y = df_subset['buy_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#TODO
def oversample_minority_class():
    pass

# normalize training data
# Scale TRAINING data with mean 0 and stdev 1
scaler = StandardScaler()
# scaler.fit(X_train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Features Shape: ", X_train_scaled.shape) 
print("Test Features Shape: ", X_test_scaled.shape) 
print("Training Labels Shape: ", y_train.shape) 


# set up keras sequental neural network
opt = Adam(lr=0.0001, decay=1e-6)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')
]

# compile model
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=METRICS)

# setup tensorbodard output
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# train model
model.fit(X_train_scaled, 
          y_train, 
          epochs=5, 
          batch_size=20, 
          callbacks=[tensorboard_callback],
          validation_data = (X_test_scaled, y_test),
          shuffle=True,
          class_weight=class_weights)

# evaluate best model
model.evaluate(X_test_scaled, y_test, batch_size=20, verbose=1)

# get predictions
y_pred = model.predict_classes(X_test_scaled)

# confusion matrix
matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print("|TP FP|")
print("|FN TN|")
print(matrix)