from Pywash2.methods.BandB.ptype.Ptype import Ptype
import pandas as pd
import numpy as np


def check_distr(df):
    '''
    Checks how many unique values there are in the categorical columns of a dataframe
    Parameters
    ----------
    df : Dataframe that needs to be checked.

    Returns
    -------
    lst : list of variables with the amount of unique values this variable takes.
    '''

    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types
    lst = []
    for key in predicted:
        if predicted[key] == 'string':
            lst.append([key, df[key].describe()['unique']])

    return lst


# path = "C:/DataScience/ptype-datasets/main/main/data.gov/3397_1"
# df = pd.read_csv(path + '/data.csv')
# a = check_distr(df)
