import pandas as pd
import tensorflow as tf
import argparse

CSV_COLUMN_NAMES = ['id', 'uid', 'gzh', 'xh', 'gz', 'middle_time', 'avg_time', 'ios', 'android', 'wifi', 'nonwifi', 'game0', 'game1', 'game2', 'game3', 'game4', 'type', 'dates']
USER_TYPE = ['ALT', 'NEW']



def load_data(y_name='type', dropped=['id', 'uid', 'dates']):
    data_dir = "model_data/bob/"

    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    csv_path = data_dir + "wl_model_data1.csv"
    dataall = pd.read_csv(csv_path, names=CSV_COLUMN_NAMES, header=0)

    dataallNew = dataall[dataall['type']==1].iloc[:100000, :]
    dataall = dataall[dataall['type']==0]

    dataall = pd.concat([dataall, dataallNew])
    dataall = dataall.sample(frac=1)

    y_all = dataall.pop(y_name)
    print(y_all.shape)
    x_all = dataall.drop(dropped, axis=1, inplace=False)
    print(x_all.shape)

    train_x = x_all.iloc[:200000, :]
    train_y = y_all.iloc[:200000]

    test_x = x_all.iloc[200000:205000, :]
    test_y = y_all.iloc[200000:205000]

    print(test_y[test_y == 1].count())
    print(test_y[test_y == 0].count())

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset




parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[5, 5], n_classes=2)

    # Train the Model.
    # classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size), steps=args.train_steps)
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, args.batch_size))

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

    print(eval_result)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predictions = classifier.predict(input_fn=lambda: eval_input_fn(test_x, labels=None, batch_size=args.batch_size))

    res = {"0-0":0, "0-1":0, "1-0":0, "1-1":0}
    for pred_dict, expect in zip(predictions, test_y):
        res[str(pred_dict['class_ids'][0]) + "-" +str(expect)] += 1

    print(res)
    #
    # # Generate predictions from the model
    # expected = ['ALT', 'NEW', 'ALT']
    # predict_x = {
    #     'gzh': [0, 1, 0],
    #     'xh': [0, 0, 0],
    #     'gz': [0, 0, 0],
    #     'deposit': [0, 0, 0],
    # }
    #
    # predictions = classifier.predict(input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))
    #
    # template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(USER_TYPE[class_id], 100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)