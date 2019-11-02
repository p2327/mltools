# coding: utf-8 
import pytest
from mltools import utilities
from mltools.utilities import *
import pandas as pd
import os

'''
_ROOT = os.path.abspath(os.path.dirname('data'))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

print(get_data('iris.csv'))
'''

# test data
iris_data = pd.read_csv('mltools/data/iris.csv')
sales_data = pd.read_csv('mltools/data/bulldozers_sales.csv', index_col=0)
currency_data = pd.read_csv('mltools/data/currency.csv', index_col=0)

# test data for make_dict
use_dict = {'a': 1, 'b': 2, 'c': 3}

# test data for fix_missing
use_df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})


def test_train_cats(): # ok
    s = utilities.train_cats(iris_data)
    assert isinstance(s['species'].dtype, pd.CategoricalDtype)


def test_add_datepart(): # ok
    s = add_datepart(sales_data, 'saledate')
    assert 'Year' in s.columns


def test_remove_column_if_label_contains(): # ok
    s = utilities.remove_column_if_label_contains(iris_data, ['length'])
    assert s.shape[1] == 3


def test_remove_row_if_column_contains():
    s = utilities.remove_row_if_column_contains(iris_data, 'sepal_width', 3.5)
    assert s['sepal_width'][0] != 3.5


# add some currency data for this
def test_cur_to_int():
    s = utilities.cur_to_int(currency_data, '£', 'Modelled_Household_median_income_estimates_2012/13')
    assert isinstance(s['Modelled_Household_median_income_estimates_2012/13'][0], np.int64)


def test_make_dict():
    s = utilities.make_dict(iris_data, 'species', 'petal_width')
    assert isinstance(s, dict)


def test_make_new_col():
    s = utilities.make_new_col(use_dict, ['a', 'c'])
    assert s[0] == 1 and s[1] == 3


def test_scale_vars():
    s = utilities.scale_vars(iris_data)
    assert isinstance(s, sklearn_pandas.DataFrameMapper)


def test_get_sample():
    s = utilities.get_sample(iris_data, 20)
    assert s.shape[0] == 20


def test_proc_df():
    s = utilities.proc_df(iris_data, 'species')
    assert s[0].shape[1] == 4


def test_fix_missing():
    s = utilities_fix_missing(use_df, 'col1', 'col1')
    assert use_df['col1'][1] == use_df['col1'].median()


def test_numericalize():
    s = utilities.numericalize(iris_data, 'species', 'setosa')
    assert s.shape[1] == 6


