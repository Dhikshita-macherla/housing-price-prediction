from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.n_features_ = None
        pass

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        This method is intended to learn any information from the input data
        necessary for the transformation.

        Parameters
        ----------
        - X : numpy.ndarray
            The input data to fit the transformer.
        - y: optional

        Returns
        -------
        - self: Object
            The fitted transformer instance.
        """
        self.n_features_ = X.shape[1]
        return self  # nothing else to do

    def transform(self, X):
        """
        Transform the data by adding derived features.

        This method creates new features by deriving them from existing columns, like rooms per household,
        population per household, and bedrooms per room. It appends these new derived features
        to the input data.

        Parameters
        ----------
        - X: numpy.ndarray
            The input data to transform.

        Returns
        -------
        - numpy.ndarray
            The transformed data with the new derived features.
        """
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]

    def inverse_transform(self, X, full_pipeline):
        """
        Inverse transforms the data based on the given pipeline.

        Used to reconstruct the original data from the transformed state.
        It applies inverse transformations from the standard scaler and one-hot encoder
        to recover the original numeric and categorical features.

        Parameters
        ----------
        - X : pandas.DataFrame
            The transformed data to be inverse-transformed.
        - full_pipeline: Pipeline Object
            The pipeline containing the transformations applied to the data.

        Returns
        --------
        - reconstructed_data: pandas.DataFrame
            A DataFrame with the inverse-transformed data, which should resemble the original data before transformation.
        """
        std_scaler = full_pipeline.named_transformers_["num"].named_steps["std_scaler"]
        inverse_num = std_scaler.inverse_transform(X.iloc[:, :-5])
        original_data = inverse_num[:, :-3]

        onehot_encoder = full_pipeline.named_transformers_["cat"]
        inverse_categ = onehot_encoder.inverse_transform(X.iloc[:, -5:])

        reconstructed_data = pd.DataFrame(
            original_data,
            columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
        )
        reconstructed_data["ocean_proximity"] = inverse_categ.flatten()

        return reconstructed_data

