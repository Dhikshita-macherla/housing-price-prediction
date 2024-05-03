import os
import unittest
import pandas as pd
from ta_lib.new_attr import FeatureExtraction


from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks([FeatureExtraction()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
