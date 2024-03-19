from typing import Callable, List, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm
from sklearn.naive_bayes import GaussianNB

DEFAULT_AGGREGATIONS = [np.mean, np.std]


def preprocess_wifi(wifi_df):
    slide_windows = wifi_df.groupby("Num_Window")

    right_rssi_values = []
    left_rssi_values = []
    occupation_label = []  # this is shared among both left&right rssi values

    for window, window_df in tqdm(slide_windows):
        right_rssi_values.append(window_df.RSSI_Right.reset_index(drop=True))
        left_rssi_values.append(window_df.RSSI_Left.reset_index(drop=True))
        occupation_label.append(window_df.is_occupied.iloc[0])

    return right_rssi_values, left_rssi_values, np.array(occupation_label)


def anomaly_detection(rssi_window, anomaly_threshold=2):
    """
    Detect anomalies in the RSSI values.
    :param anomaly_threshold: Threshold for selecting anomaly.
    :param rssi_window: RSSI values of a window.
    :return: rssi_window without anomalies in the RSSI values.
    """
    sigma = np.std(rssi_window)
    median = np.median(rssi_window)
    rssi_window[((rssi_window - median) / sigma) > anomaly_threshold] = None
    return rssi_window


class BaselineModel:
    def __init__(self, anomaly_threshold=None, model=None, aggregations: List[Callable] = DEFAULT_AGGREGATIONS):
        self.anomaly_threshold = anomaly_threshold
        self.aggregations = aggregations
        if model is not None:
            self.model = model
        else:
            log_clf = LogisticRegression()
            bayes_net = GaussianNB()
            svm_clf = SVC(kernel="linear")
            # knn = KNeighborsClassifier(n_neighbors=5)
            self.model = VotingClassifier(
                estimators=[('lr', log_clf), ('bn', bayes_net), ('svc', svm_clf)], voting='hard')

    def fit(self, rssi_values_df, occupation_label, sample_weight: Union[list, np.ndarray, str] = None):
        """
        Fit the model to the given data.
        :param rssi_values_df: dataframe of RSSI data - dataframe of windows of RSSI values
                               (e.g, left and/or right antennas).
        :param occupation_label: list of labels for the occupation of the room.
        :param sample_weight: sample weights for the training data. If None, then samples are equally weighted.
        """
        features = self.create_features(rssi_values_df.copy())
        pos_count = sum(occupation_label)
        neg_count = len(occupation_label) - pos_count
        if isinstance(sample_weight, str):
            if sample_weight == "balanced":
                sample_weight = np.array([1 if label == 1 else pos_count / neg_count for label in occupation_label])
            else:
                raise ValueError("Invalid value for sample_weight. Expected 'balanced' or an array-like of shape "
                                 f"(n_samples,) but got {sample_weight}.")
        self.model.fit(features, occupation_label, sample_weight=sample_weight)

    def predict(self, rssi_values_df: pd.DataFrame):
        """
        Predict the occupation of the room.
        :param rssi_values_df: dataframe of RSSI data - dataframe of windows of RSSI values
                               (e.g, left and/or right antennas).
        :return: Predictions for the occupation of the room.
        """
        features = self.create_features(rssi_values_df.copy())
        predictions = self.model.predict(features)
        return predictions

    def predict_proba(self, rssi_values_df: pd.DataFrame):
        """
        Predict probabilities of the occupation of the room.
        :param rssi_values_df: dataframe of RSSI data - dataframe of windows of RSSI values
                               (e.g, left and/or right antennas).
        :return: Predictions for the occupation of the room.
        """
        features = self.create_features(rssi_values_df.copy())
        predictions = self.model.predict_proba(features)
        return predictions

    def create_features(self, rssi_values_df: pd.DataFrame):
        """
        Create features from the given RSSI values.
        :param rssi_values_df: dataframe of RSSI data - dataframe of windows of RSSI values
                               (e.g, left and/or right antennas).
        :param aggregations: list of aggregation functions to apply on the RSSI values.
        """
        feature_names = []

        for col in rssi_values_df.columns:
            if self.anomaly_threshold is not None:
                rssi_values_df[col] = rssi_values_df[col].apply(
                    lambda x: anomaly_detection(x, self.anomaly_threshold)).values
            for func in self.aggregations:
                feature_names.append(f'rssi_{str(func.__name__)}_{col}')
                rssi_values_df[f'rssi_{str(func.__name__)}_{col}'] = rssi_values_df[col].apply(func)
        return rssi_values_df[feature_names].fillna(0)
