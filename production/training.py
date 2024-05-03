"""Processors for the model training step of the worklow."""
import logging
import logging.config
import os.path as op
from sklearn.pipeline import Pipeline
import numpy as np
from ta_lib.core.api import (
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)
from sklearn.model_selection import GridSearchCV
import ta_lib.eda.api as eda
import pandas as pd
from ta_lib.regression.api import SKLStatsmodelOLS
import ta_lib.reports.api as reports
logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """
    Train a regression model.

    Parameters:
    - context: The context object containing information about the execution environment.
    - params: Additional parameters or configuration settings for scoring the model.

    Returns:
    - None
    input_features_ds: Path to input features dataset
    input_target_ds: Path to input target dataset
    artifacts_folder: Path to folder to store artifacts

    """
    logger.info("Training a regression model")
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    logger.info("Loading training dataset")
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    logger.info("Performing EDA")
    out_plot = eda.get_density_plots(pd.DataFrame(train_X))
    print(out_plot)
    reports.create_report({'univariate': out_plot}, name='production/reports/feature_analysis_univariate')
    reports.feature_analysis(train_X, 'production/reports/feature_analysis_report.html')

    # create reports as needed
    cols = train_X.columns.to_list()
    all_plots = {}
    for ii, col1 in enumerate(cols):
        for jj in range(ii + 1, len(cols)):
            col2 = cols[jj]
            out = eda.get_bivariate_plots(train_X, x_cols=[col1], y_cols=[col2])
            all_plots.update({f'{col2} vs {col1}': out})

    reports.create_report(all_plots, name='production/reports/feature_analysis_bivariate')
    reports.feature_interactions(train_X, 'production/reports/feature_interaction_report.html')
    reports.data_exploration(train_X, train_y, 'production/reports/data_exploration_report.html', y_continuous=True)
    logger.info("Reports generated and saved successfully")

    # create training pipeline
    logger.info("Creating training pipeline for Linear Regression")
    lin_reg_ppln = Pipeline([
        ('linreg_estimator', SKLStatsmodelOLS())
    ])
    lin_reg_ppln.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    logger.info("Saved fitted training pipeline")
    save_pipeline(lin_reg_ppln, op.abspath(op.join(artifacts_folder, "lin_reg_pipeline.joblib")))
    logger.info("Linear Regression done")
    print("LR done")

    logger.info("Creating training pipeline for decision tree")
    from sklearn.tree import DecisionTreeRegressor
    dtree_reg_ppln = Pipeline([
        ('dtreereg_estimator', DecisionTreeRegressor())
    ])
    dtree_reg_ppln.fit(train_X, train_y.values.ravel())
    logger.info("Saved fitted training pipeline")
    save_pipeline(dtree_reg_ppln, op.abspath(op.join(artifacts_folder, "dtree_reg_pipeline.joblib")))
    logger.info("Decision Tree Regression done")
    print("DT done")

    logger.info("Model tuning")
    logger.info("Randomized Search CV for Random forest regressor")
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import randint
    forest_reg = RandomForestRegressor(random_state=params["rand_regressor"]["random_state"])
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    rand_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=params["random_forest"]["n_iter"],
        cv=params["random_forest"]["cv"],
        scoring=params["random_forest"]["scoring"],
        random_state=params["random_forest"]["random_state"],
    )
    rand_search.fit(train_X, train_y.values.ravel())
    print("Best params from Randomized Search CV", rand_search.best_params_)
    cvres = rand_search.cv_results_
    for mean_score, params_cv in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params_cv)

    feature_importances = rand_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, train_X.columns), reverse=True)
    final_model_rand = rand_search.best_estimator_
    print("Best estimator for Randomized Search CV: ", final_model_rand)
    logger.info("Saved fitted training pipeline")
    save_pipeline(final_model_rand, op.abspath(op.join(artifacts_folder, "rand_search.joblib")))
    logger.info('Randomized Search CV done')
    print("Rand_search")
    print(params)
    logger.info("Grid Search CV for Random forest regressor")
    param_grid = params["grid_search"]["param_grid"]

    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=params["grid_search"]["cv"],
        scoring=params["grid_search"]["scoring"],
        return_train_score=params["grid_search"]["return_train_score"],
    )

    grid_search.fit(train_X, train_y.values.ravel())
    print("Best params for Grid Search CV: ", grid_search.best_params_)
    cvres = grid_search.cv_results_
    for mean_score, params_cv in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params_cv)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, train_X.columns), reverse=True)
    final_model_grid = grid_search.best_estimator_
    print("Best estimator for Grid Search CV: ", final_model_grid)
    logger.info("Saved fitted training pipeline")
    save_pipeline(final_model_grid, op.abspath(op.join(artifacts_folder, "grid_search.joblib")))
    logger.info("Grid Search done")
    logger.info("Training done")
    print("Training done")
