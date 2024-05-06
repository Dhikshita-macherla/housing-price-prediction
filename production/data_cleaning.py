"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import yaml
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import logging
from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)
from scripts import binned_selling_price
'''with open('production/conf/logging/production.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

logging.config.dictConfig(config)'''
logger = logging.getLogger(__name__)

@register_processor("data-cleaning", "housing")
def clean_house_table(context, params):
    """
    Cleans the housing data table.

    Parameters:
    - context: Context object containing information about the execution environment.
    - params: Additional parameters or configuration settings for the cleaning process.

    Returns:
    - housing_df_clean: The cleaned housing dataset.

    Description:
    This function cleans the housing data table by performing the following steps:
    1. Loads the input dataset from the path specified by "raw/housing".
    2. Identifies string columns in the dataset, excluding specific numerical columns.
    3. Converts specific numerical columns to integer type.
    4. Cleans string columns using a custom cleaning function called string_cleaning.
    5. Saves the cleaned dataset to the path specified by "cleaned/housing".
    6. Returns the cleaned housing dataset.

    """
    logger.info("Cleaning housing dataset")
    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"
    logger.info("Loading dataset from raw/housing")

    # load dataset
    housing_df = load_dataset(context, input_dataset)
    logger.info("Identifying string columns")
    str_cols = list(
        set(housing_df.select_dtypes('object').columns.to_list())
        - set(['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'])
    )
    logger.info("Cleaning dataset...")
    housing_df_clean = (
        housing_df
        # set dtypes
        .change_type(['housing_median_age', 'total_rooms', 'population', 'households', 'median_house_value'], np.int64)

        .transform_columns(str_cols, string_cleaning, elementwise=False)
    )

    # save the dataset
    logger.info("Saving cleaned dataset to cleaned/housing")
    save_dataset(context, housing_df_clean, output_dataset)
    logger.info("Housing dataset cleaned and saved successfully")
    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """
    Split the housing table into train and test datasets.

    Parameters:
    - context: Context object containing information about the execution environment.
    - params: Additional parameters or configuration settings for the cleaning process.

    Return: None

    Input dataset: "cleaned/housing"
    Output train features: "train/housing/features"
    Output train target: "train/housing/target"
    Output test features: "test/housing/features"
    Output test target: "test/housing/target"

    Description:
    - Loads the cleaned housing dataset from the specified input dataset path.
    - Splits the data using StratifiedShuffleSplit into train and test sets.
    - Save the features to the specified output path for train and test features.
    - Save the target to the specified output path for train and test target.
    """
    logger.info("Housing dataset into train and test sets")
    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    logger.info("Loading dataset from cleaned/housing")
    housing_df_processed = load_dataset(context, input_dataset)

    # split the data
    logger.info("Splitting the dataset into train and test")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, by=binned_selling_price
    )

    # split train dataset into features and target
    logger.info("Splitting train dataset into features and target")
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    logger.info("Saving the train datasets")
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    logger.info("Splitting test dataset into features and target")
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    logger.info("Saving the test datasets")
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
    logger.info("Completed cleaning and splitting housing dataset")


print("Completed cleaning and splitting housing dataset")