"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op
import pandas as pd
from ta_lib.new_attr import FeatureExtraction
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from ta_lib.data_processing.api import Outlier
from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    register_processor,
    save_dataset,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """
    Transform dataset to create training datasets.
    Parameters:
    - context: Context object containing information about the execution environment.
    - params: Additional parameters or configuration settings for the cleaning process.

    Returns: None

    input_features_ds: Path to input features dataset
    input_target_ds: Path to input target dataset
    artifacts_folder: Path to folder to store artifacts
    """
    logger.info("Transforming dataset to create training datasets")
    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    logger.info("Loading datasets")
    housing_df = load_dataset(context, 'raw/housing')
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)
    test_X = load_dataset(context, "test/housing/features")
    test_y = load_dataset(context, "test/housing/target")
    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    logger.info("Treating outliers")
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )
    logger.info("Defining pipelines for preprocessing")

    def pipelinesIngestion(housing):
        housing = housing.drop("median_house_value", axis=1)
        X_num = housing.drop("ocean_proximity", axis=1)
        num_attribs = list(X_num)
        cat_attribs = ["ocean_proximity"]

        full_pipeline_new = ColumnTransformer([("num", Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', FeatureExtraction()),
            ('std_scaler', StandardScaler()),]), num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),])
        return full_pipeline_new, housing
    full_pipeline, housing_df = pipelinesIngestion(housing_df)

    # housing_df_copy = housing_df.copy()
    print("Original train X before transforming: ", train_X)

    logger.info("Transforming features")
    train_X = get_dataframe(
        full_pipeline.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(full_pipeline),
    )
    train_X.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'population_per_household', 'bedrooms_per_room', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
    print("After transforming: ", train_X)

    logger.info("Transforming target")
    test_X = get_dataframe(
        full_pipeline.transform(test_X),
        get_feature_names_from_column_transformer(full_pipeline)
    )
    test_X.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'population_per_household', 'bedrooms_per_room', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']

    logger.info("Saving pipeline and datasets")
    save_pipeline(full_pipeline, op.abspath(op.join(artifacts_folder, "features.joblib")))
    save_dataset(context, train_X, input_features_ds)
    save_dataset(context, train_y, input_target_ds)
    save_dataset(context, test_X, "test/housing/features")
    save_dataset(context, test_y, "test/housing/target")

    feature_extraction = FeatureExtraction()
    reconstructed_data = feature_extraction.inverse_transform(train_X, full_pipeline)
    print(reconstructed_data)
    logger.info("Feature engineering done")


print("Feature engineering done")