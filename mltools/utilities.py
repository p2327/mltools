import functools
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

import sklearn
import sklearn_pandas as skp
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import time
from typing import Callable, List, Union, Tuple, Any
import warnings


def timer(f: Callable) -> Callable:
    """
    Timer decorator implementation.
    Print the runtime of decorated functions.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        end = time.perf_counter()
        run_time = end - start
        print(f"{f.__name__!r} finished in {run_time:4f} seconds")
        return result
    return wrapper


def train_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change any columns of strings in a pandas dataframe to a column of
    categorical values.

    Parameters:
    -----------
    df: a pandas dataframe.

    Example usage:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> df.col2.dtype
    str
    >>> train_cats(df)
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> df.col2.dtype
    category
    """
    for label, content in df.items():
        if is_string_dtype(content):
            df[label] = content.astype('category').cat.as_ordered()
    return df


def add_datepart(df: pd.DataFrame, date_field_name: str, drop: bool = True,
                 time: bool = False) -> pd.DataFrame:
    """
    Helper function that manipulates datetime data.
    Adds columns relevant to discrete properties of a date.
    """
    # make target date column generic
    df.rename(columns={date_field_name: 'date_'}, inplace=True)
    date_field_name = [i for i in df.columns if i == 'date_'][0]
    # initialise variables
    field = df[date_field_name]
    field_dtype = field.dtype

    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64

    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field_name] = field = pd.to_datetime(
                                            field,
                                            infer_datetime_format=True)

    # leaves dp and add attributes
    # targ_pre = re.sub('[Dd]ate$', '', date_field_name)
    attributes = ['year',
                  'month',
                  'week',
                  'day',
                  'dayofweek',
                  'dayofyear',
                  'is_month_end',
                  'is_month_start',
                  'is_quarter_end',
                  'is_quarter_start',
                  'is_year_end',
                  'is_year_start']

    if time:
        attributes = attributes + ['hour', 'minute', 'second']

    # creates new features with pattern+attribute
    for attribute in attributes:
        df[date_field_name + attribute] = getattr(field.dt, attribute.lower())
    # adds the number of days since the first date in the dataset
    df[date_field_name + 'elapsed'] = field.astype(np.int64) // 10 ** 9

    if drop:
        df.drop(date_field_name, axis=1, inplace=True)

    return df


def remove_column_if_label_contains(df: pd.DataFrame,
                                    cols: List[str]) -> pd.DataFrame:
    """
    Removes features in a dataframe if a feature name
    is partially matching any names in cols.
    """
    for name in cols:
        df.drop([col for col in df.columns if name in col],
                axis=1, inplace=True)
    return df


# could add print for shape before and after
def remove_row_if_column_contains(df: pd.DataFrame,
                                  label: str,
                                  value: Union[str, int, float]
                                  ) -> pd.DataFrame:
    """
    Removes a row from a dataframe where value is in column.
    """
    index_names = df[df[label] == value].index
    df.drop(index_names, inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df


def cur_to_int(df: pd.DataFrame, symbol: str, column: str) -> pd.DataFrame:
    """
    Convert a currency from str to int and removes its symbol.
    """
    loc = df.columns.get_loc(column)
    clean_col = df[column].replace(f'[{symbol},]',
                                   '', regex=True).astype(np.int64)
    df.drop(column, axis=1, inplace=True)
    df.insert(loc, column, clean_col)
    return df


def make_dict(df: pd.DataFrame, key_column: str, value_column: str) -> dict:
    """
    Creates a dictionary matching two columns in a dataframe.
    """
    return dict(zip(df[key_column].values, df[value_column].values))


# encode new columns matching dictionary of values to list from dataframe
def make_new_col(d: dict, keys: List[str]) -> List:
    """
    Extract values from a dictionary in a match exist in keys passed.
    This can be used to add new columns to a dataframe.
    """
    return [d[key] for key in keys]


def scale_vars(df: pd.DataFrame,
               mapper: DataFrameMapper = None) -> skp.DataFrameMapper:
    """
    Returns a mapper to scale variables.
    """
    warnings.filterwarnings('ignore',
                            category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        # apply standardscaler to columns
        map_f = [(
            [column], StandardScaler()
            ) for column in df.columns if is_numeric_dtype(df[column])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper


def get_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Returns a random dataframe subset of n rows.
    It does not reset the index.
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()


def proc_df(df: pd.DataFrame, y_fld: str = None,
            skip_flds: List[str] = None, ignore_flds: List[str] = None,
            do_scale: bool = False, na_dict: dict = None,
            preproc_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
            max_n_cat: int = None, subset: int = None,
            mapper=None) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Takes a dataframe and:
        Splits off the response variable
        Changes the nonumeric df.dtypes into numeric.
    For each column which is not in skip_flds or in ignore_flds,
    NaN values are replaced by the median value of the column.
    """
    if not ignore_flds:
        ignore_flds = []

    if not skip_flds:
        skip_flds = []

    if subset:
        df = get_sample(df, subset)
    else:
        df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn:
        preproc_fn(df)

    if y_fld is None:
        y = None
    else:
        if not is_numeric_dtype(df[y_fld]):
            df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None:
        na_dict = {}
    else:
        na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()

    for label, content in df.items():
        na_dict = fix_missing(df, content, label, na_dict)

    if len(na_dict_initial.keys()) > 0:
        df.drop([
            a + '_na' for a in list(set(na_dict.keys()) -
                                    set(na_dict_initial.keys()))
            ], axis=1, inplace=True)

    if do_scale:
        mapper = scale_vars(df, mapper)

    for label, content in df.items():
        numericalize(df, content, label, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]

    if do_scale:
        res = res + [mapper]

    return res


def fix_missing(df: pd.DataFrame, col: pd.Series,
                target_label: str, na_dict: dict = None) -> dict:
    """
    Replaces na values with median if data is numeric.
    Adds _na suffix columns where True means a NaN was replaced.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, 'col1', 'col1')
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False
    """
    # assumes a numerica dtype
    if na_dict is None:
        na_dict = {}
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (target_label in na_dict):
            df[target_label+'_na'] = pd.isnull(col)
            filler = na_dict[target_label] if target_label in na_dict else col.median()
            df[target_label] = col.fillna(filler)
            na_dict[target_label] = filler
    return na_dict


def numericalize(df: pd.DataFrame, col: pd.Series,
                 target_label: str, max_n_cat: int = None) -> pd.DataFrame:
    """
    Converts non numeric values to numeric.
    """
    # codes+1 is to have NaN values set to 0 and not -1
    if not is_numeric_dtype(col) and (max_n_cat is None or len(col.cat.categories) > max_n_cat):
        df[target_label] = pd.Categorical(col).codes+1
    return df


def dedupe(data: List[Any]) -> List[Any]:
    data_set = set(tuple(x) for x in data)
    data_deduped = [list(x) for x in data_set]
    return data_deduped


# TODO
def split_vals(df, n):
    '''
    Returns df copy of up to n and from n to end of the rows
    df[:n] for training and df[n:] for validation.
    '''
    return df[:n].copy(), df[n:].copy()


def peek(df):
    with pd.option_context("display.max_colwidth", 20):
        info = pd.DataFrame()
        info['sample'] = df.iloc[0]
        info['data type'] = df.dtypes
        info['percent missing'] = df.isnull().sum()*100/len(df)
        return info.sort_values('data type')


def rf_feat_importance(m, df):
    return pd.DataFrame({
        'cols': df.columns,
        'imp': m.feature_importances_
        }).sort_values('imp', ascending=False)


def plot_fi(features):
    return features.plot('cols',
                         'imp',
                         'barh',
                         figsize=(8, 6),
                         colormap='RdBu',
                         legend=False
                         )
