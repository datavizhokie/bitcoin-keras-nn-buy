import snowflake.connector
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json

with open('configs.json') as f:
  configs = json.load(f)

signal = 'buy_binary'
date_time_format = '%Y-%m-%d %H:%M'


def pull_snowflake_data(table, signal):

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

    query = f'''
    select * from TRADING_DB.BTC_USD.{table};
    '''
    cur = conn.cursor().execute(query)
    df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
    print(f'There are {len(df)} records read from Snowflake table {table}')

    # Create modeling signal
    df[signal] = np.where(df.ORDER_SIGNAL=='buy', 1, 0)

    return df



def subset_for_datemin(df, date_field, datemin, signal):

    ''' Selects the minimum date for training;
        the maxium date will be the latest date
        from Snowflake 
    '''


    datetime_min = df[date_field].min().strftime(date_time_format)
    datetime_max = df[date_field].max().strftime(date_time_format)

    print("Raw Date Metrics:")
    print("Date Min: ", datetime_min)
    print("Date Max: ", datetime_max)

    df_subset = df[df.TIME >= datemin]
    print(f"There are {len(df_subset)} records in the subset with min date {datemin} and max date {datetime_max}")

    print("Class Counts of Training Subset:")
    df_subset.groupby(signal)['SEQUENCE'].count()

    return df_subset, datetime_min, datetime_max


# Create weights to combat class imbalance
def create_class_imb_weights(df, num_outcomes, response):
    ''' Model can be trained with class weights to combat class imbalance '''

    no_pos_count, pos_count = np.bincount(df[response])
    total_count = len(df[response])

    weight_neg = (1 / no_pos_count) * (total_count) / num_outcomes
    weight_pos = (1 / pos_count) * (total_count) / num_outcomes

    #TODO: enumerate for >2 outcomes
    class_weights = {0: weight_neg, 1: weight_pos}
    print("Class Weights: ", class_weights)


def choose_predictors(df_subset, fields_to_remove):

    ''' Select predictor fields by feeding in fields to exclude;
        Tensorflow does not like string or date fields - use
        categorical encoding and/or datetime -> ordinal if needed
    '''

    # Choose predictor fields
    fields = list(df_subset.columns.values)
    print("All Fields:")
    print(fields)

    fields_pred = [e for e in fields if e not in fields_to_remove]

    print('Fields to be used as predictors:')
    print(fields_pred)

    pred_dim = len(fields_pred)

    return fields_pred, pred_dim


def setup_train_and_test(df_subset, test_size, fields_pred, signal):

    # training and test sets
    X = df_subset[fields_pred]
    y = df_subset[signal]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test


def oversample_minority_class(fields_pred, signal, X_train, y_train):
    ''' To remedy the class imbalance (outcome "buy" being rare),
        oversample the "buy" outcome 
    '''

    X = pd.concat([X_train, y_train], axis=1)
    no_buy = X[X[signal] == 0]
    buy = X[X[signal] == 1]

    buy_upsampled = resample(buy,
                             replace=True,
                             n_samples=len(no_buy),
                             random_state=12345)

    train_resmpled = no_buy.append(buy_upsampled)
    print("Class Counts After Oversampling:")
    print(train_resmpled.groupby(signal)[signal].count())

    X_train = train_resmpled[fields_pred]
    y_train = train_resmpled[signal]

    return X_train, y_train


def norm_train_data(X_train, X_test, y_train):
    ''' Tensorflow requires normalized data for training '''
    # Scale TRAINING data with mean 0 and stdev 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Features Shape: ", X_train_scaled.shape) 
    print("Test Features Shape: ", X_test_scaled.shape) 
    print("Training Labels Shape: ", y_train.shape) 

    return X_train_scaled, X_test_scaled


# set up keras sequental neural network
def compile_model(lr, decay, pred_dim):
    ''' Set up neural network layers,
        choose evaluation metrics,
        compile model '''

    opt = Adam(lr=lr, decay=decay)

    # For binary classification, use Sigmoid activation as the last layer
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(pred_dim,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    METRICS = [
        keras.metrics .TruePositives(name='tp'),
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

    return model, tensorboard_callback, lr, decay


def train_neural_network(model, X_test_scaled, X_train_scaled, y_train, y_test, lr, decay, epochs, batch_size, tensorboard_callback, datetime_min, datetime_max):
    ''' Train model across specified epochs and batch size;
        Viz Loss across epoch;
        Evaluate best overall model from training
    '''

    print(f"Learning Rate is {lr}")
    print(f"Decay is {decay}")
    print(f"Training for {epochs} epochs...")

    history = model.fit(X_train_scaled, 
                        y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[tensorboard_callback],
                        validation_data = (X_test_scaled, y_test),
                        shuffle=True)
                        #class_weight=class_weights)

    # evaluate best model
    score = model.evaluate(X_test_scaled, y_test, batch_size=10, verbose=1)

    print('Test loss:', round(score[0], 3))
    print('Test accuracy:', round(score[5], 3))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f"Model Loss: Scope {datemin}-{datetime_max}")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"Model Loss: lr={lr}, decay={decay}, epochs={epochs}, batch_size={batch_size}; Scope {datetime_min}-{datetime_max}.png", format="png")
    #plt.show(block=False)

    # get predictions
    y_pred = model.predict_classes(X_test_scaled)

    # confusion matrix
    matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print("|TP FP|")
    print("|FN TN|")
    print(matrix)

    return y_pred


def main():

    df = pull_snowflake_data('TRADE_PROFIT_SIGNALS_BY_HOUR', signal)
    df_subset, datetime_min, datetime_max = subset_for_datemin(df, 'TIME','2020-09-10', signal)
    class_weights = create_class_imb_weights(df_subset, 2, signal)

    fields_pred, pred_dim = choose_predictors(df_subset, (signal,'PROFIT_SIGNAL','SEQUENCE',
                                            'ORDER_SIGNAL','TIME','PREV_MINUTE','PREV_HOUR'))

    X_train, X_test, y_train, y_test = setup_train_and_test(df_subset, 0.3, fields_pred, signal)
    X_train, y_train = oversample_minority_class(fields_pred, signal, X_train, y_train)
    X_train_scaled, X_test_scaled = norm_train_data(X_train, X_test, y_train)
    model, tensorboard_callback, lr, decay = compile_model(0.001, 1e-6, pred_dim)
    y_pred = train_neural_network(model, X_test_scaled, X_train_scaled, y_train, y_test, lr, decay, 20, 10, tensorboard_callback, datetime_min, datetime_max)


if __name__== "__main__" :
    main()