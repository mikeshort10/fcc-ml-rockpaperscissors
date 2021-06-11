# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import tensorflow as tf
import pandas as pd


def input_fn(data_df, label_df, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

    if training:
        dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def last(l):
    return l[len(l) - 1]


def get_prev_round(prev_play, my_prev_play, two_rounds_ago):
    return [prev_play, my_prev_play, *two_rounds_ago[0:len(two_rounds_ago) - 2]]


def rps_to_num(rps):
    if rps == "R":
        return 0
    elif rps == "P":
        return 1
    elif rps == "S":
        return 2
    return -1


def predict_input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


def get_last(l, amount=50):
    return l[::-1][0:amount][::-1]


def create_classifier():

    classifier = None

    def get_classifier(game_history, columns, train=True, reset=False):
        nonlocal classifier
        if train == False and classifier is not None:
            return classifier
        dftrain = pd.DataFrame(game_history)
        dftrain.columns = columns
        y_train = dftrain.pop("one")
        feature_columns = []
        for key in dftrain.keys():
            feature_columns.append(
                tf.feature_column.numeric_column(key=key))

        if reset:
            print("💥 🤖 resetting classifier")
            classifier = tf.estimator.DNNClassifier(
                feature_columns=feature_columns, n_classes=3, hidden_units=[50, 30],
                # model_dir="/var/folders/hr/kksw_vkd4tgc2rsy_627h2x40000gn/T/tmp1mqlx4yx"
            )
        print("Starting training session", len(game_history))

        classifier.train(
            input_fn=lambda: input_fn(
                get_last(dftrain), get_last(y_train), training=True),
            steps=5000)
        print("Finished training session")
        return classifier
    return get_classifier


def player_setup():

    history_length = 10

    game_history = []

    training_freq = 50

    CATEGORICAL_COLUMNS = ["one", "my_one", "two", "my_two", "three",
                           "my_three", "four", "my_four", "five", "my_five"]

    expected = ["R", "P", "S"]

    my_prev_play = "R"

    has_full_game_history = False

    reset_classifier = True

    get_classifier = create_classifier()

    print("🖖 Hello, I'm Nemo. Would you like to play a game?")

    # TODO: Try with more history

    def player(prev_play):
        nonlocal game_history
        nonlocal my_prev_play
        nonlocal reset_classifier
        nonlocal has_full_game_history

        if prev_play == "":
            has_full_game_history = False
            reset_classifier = True
            game_history = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
            return my_prev_play

        prev_round = get_prev_round(rps_to_num(
            prev_play), rps_to_num(my_prev_play), last(game_history))

        game_history.append(prev_round)

        should_initiate_ml = has_full_game_history == False and len(
            game_history) == history_length + 2

        # remove incomplete datasets once we've got enough data
        if len(game_history) <= history_length + 1:
            return my_prev_play
        elif should_initiate_ml:
            game_history = game_history[history_length + 1:]
            has_full_game_history = True

        should_retrain = should_initiate_ml or len(
            game_history) % training_freq == 0

        classifier = get_classifier(reset=reset_classifier,
                                    game_history=game_history, columns=CATEGORICAL_COLUMNS, train=should_retrain)
        if reset_classifier:
            reset_classifier = False
        predict_x = {
            "my_one": [prev_round[0]],
            "two": [prev_round[1]],
            "my_two": [prev_round[2]],
            "three": [prev_round[3]],
            "my_three": [prev_round[4]],
            "four": [prev_round[5]],
            "my_four": [prev_round[6]],
            "five": [prev_round[7]],
            "my_five": [prev_round[8]],
        }

        predictions = classifier.predict(
            input_fn=lambda: predict_input_fn(predict_x))

        for pred_dict, expec in zip(predictions, expected):
            class_id = pred_dict['class_ids'][0]
            my_prev_play = expected[class_id]

            return my_prev_play

    return player


player = player_setup()

#! P1
# ? Player 1 win rate: 46.65809768637532%

# ? Player 1 win rate: 49.308176100628934%

# ? Player 1 win rate: 63.48808030112924%

# Player 1 win rate: 63.76988984088128%

#! P2

# Player 1 win rate: 0.1002004008016032%

#! P3

# Player 1 win rate: 33.691275167785236%

#! P4

# Player 1 win rate: 33.819628647214856%
