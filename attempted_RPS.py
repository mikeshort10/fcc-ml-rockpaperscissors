# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        print('hello')
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        print(ds)
        print('hello')
        if shuffle:
            ds = ds.shuffle(1000)
        print('hello')

        ds = ds.batch(batch_size).repeat(num_epochs)
        print(ds)
        return ds
    return input_function


def last(l):
    return l[len(l) - 1]


def get_prev_round(prev_play, two_rounds_ago):
    return [prev_play, *two_rounds_ago[0:len(two_rounds_ago) - 1]]


def player_setup():

    history_length = 5

    game_history = [["", "", "", "", ""]]

    def player(prev_play):

        nonlocal game_history

        prev_round = get_prev_round(prev_play, last(game_history))

        game_history.append(prev_round)

        guess = "R"

        # remove incomplete datasets once we've got enough data

        if len(game_history) <= history_length + 100:
            return guess
        elif len(game_history) == history_length + 101:
            game_history = game_history[history_length + 1:]

        df = pd.DataFrame(game_history)
        df.columns = ["1", "2", "3", "4", "5"]

        df_train = df.loc[:, df.columns != "1"]
        y_train = df.loc[:, "1"]

        df_eval = pd.DataFrame(prev_round[1: history_length])
        # this is an arbitrary value
        y_eval = pd.DataFrame(["R"])

        feature_columns = []
        for index in df_train.columns:
            vocabulary = df_train[index].unique()
            feature_columns.append(
                tf.feature_column.categorical_column_with_vocabulary_list(index, vocabulary))

        train_input_fn = make_input_fn(df_train, y_train)

        eval_input_fn = make_input_fn(df_eval, y_eval)

        print(df_train.shape, y_train.shape)

        linear_est = tf.estimator.LinearClassifier(
            feature_columns=feature_columns)
        print(df_eval.shape, y_eval.shape)
        print(df.info(), feature_columns)

        linear_est.train(input_fn=train_input_fn)

        print("here")

        # result = list(linear_est.predict(eval_input_fn))

        # print(result[0]['probabilities'])

        return guess

    return player


player = player_setup()
