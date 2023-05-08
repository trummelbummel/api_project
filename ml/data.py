'''
Code to preprocess and clean data
for the Census Data Salaray prediction.
'''
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def clean_data(data, label='salary'):
    """
    Takes the input data and cleans the spaces
    from the dataframe.
    Furthermore balance the class imbalance,
    to improve performance.
     data : pd.DataFrame
        Dataframe containing the features and label.
    label: str Target to use for classification.
    """
    # remove duplicate rows, and strip spaces away.
    data.columns = [
        colname.strip(' ').replace('-', '_') for colname in data.columns]
    data = data.applymap(lambda element:
                         element.strip(' ') if isinstance(element, str)
                         else element)
    data = data.drop_duplicates()
    return data


def process_data(
    data, categorical_features=[], label=None,
        training=True, encoder=None, lb=None
):
    """ Process the data used in
    the machine learning pipeline.

    Processes the data using one hot encoding for the
    categorical features and a
    label binarizer for the labels.
    This can be used in either training or
    inference/validation.

    Note: depending on the type of
    model used, you may
    want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
         Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the
        categorical features (default=[])
    label : str
        Name of the label column in `X`. If None,
        then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X_features : np.array
        Processed data.
    labels : np.array
        Processed labels if labeled=True, otherwise
        empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True,
        otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True,
         otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        labels = data[label]
        features = data.drop([label], axis=1)

    else:
        features = data
        labels = np.array([])

    X_categorical = features[categorical_features].values
    X_continuous = features.drop(*[categorical_features],
                                 axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False,
                                handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        labels = lb.fit_transform(labels.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            labels = lb.transform(labels.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X_features = np.concatenate([X_continuous.values,
                                 X_categorical], axis=1)
    return X_features, labels, encoder, lb
