import pandas as pd
import numpy as np
from scipy import stats
from Pywash2.methods.BandB.ptype.Ptype import Ptype
from Pywash2.methods.BandA.OutlierDetector import estimate_contamination, identify_outliers
import seaborn as sns
from pyod.models.knn import KNN as knn
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


def interpretable_values(df):
    '''
    Checks whether the data is documented or not.

    Parameters
    ----------
    df : the dataframe that needs analyzing.

    user_input : the user tells us whether the data has been documented, partially documented, or not documented.

    Returns
    -------
    quality_measure : The quality measure of documentation.
    '''
    ##TODO
    #userinput whether the data is documented, partially documented or not documented.
    try:
        if user_input == 'not documented':
            quality_measure = 0
        elif user_input == 'partially documented':
            quality_measure = 0.5
        if user_input == 'documented':
            quality_measure = 1
    except:
        quality_measure = 0
    return quality_measure

def feature_scaling(df):
    '''
    Performs a kolmogorov-smirnov on all the columns that were predicted to be numerical. Then calculates which
    percentage of these columns is normally distributed.

    Parameters
    ----------
    df : the dataframe that needs analyzing.


    Returns
    -------
    quality_measure : The percentage of normalized continuous columns.
    '''
    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types
    count_normal_vars = 0
    count_continuous_vars = 0
    for key in predicted:
        # print(key, predicted[key])
        if predicted[key] == 'int' or predicted[key] == 'float':

            try:
                pd.to_numeric(df[key])
                count_continuous_vars += 1
                if stats.kstest(df[key], 'norm').pvalue <= 0.05:
                    continue
                else:
                    count_normal_vars += 1

                # null-hypothesis is no difference. p-value <= 0.05: not normal.
            except:
                print('Column {} could not be transformed to numeric.'.format(key))

    if count_continuous_vars > 0:
        quality_measure = count_normal_vars / count_continuous_vars * 100
    else:
        quality_measure = 1


    return quality_measure

def outlier_detection(df):
    '''
    Performs a contamination check and an outlier detection to see what percentage of the data is an outlier.

    Parameters
    ----------
    df : the dataframe that needs analyzing.


    Returns
    -------
    quality_measure : The percentage of outliers in the data.
    '''
    contamination = estimate_contamination(df)
    features = df.columns
    outliers = identify_outliers(df, features=features, contamination=contamination)[0]
    perc = outliers['prediction'].sum()/df.shape[0] * 100
    if perc <= 1:
        quality_measure = 1
    elif perc > 1 and perc <= 5:
        quality_measure = 0.75
    elif perc > 5 and perc <= 10:
        quality_measure = 0.5
    elif perc > 10 and perc <= 20:
        quality_measure = 0.25
    else:
        quality_measure = 0
    return quality_measure

def feature_selection(df, target):
    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types
    count_normal_vars = 0
    count_continuous_vars = 0
    features = []
    for key in predicted:
        # print(key, predicted[key])
        if predicted[key] == 'int' or predicted[key] == 'float':
            features.append(key)
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    x = pd.DataFrame(x)
    x.columns = features


    X = x.drop(target, 1)  # Feature Matrix
    y = x[target]  # Target Variable

    # no of features
    nof_list = np.arange(1, len(features))
    high_score = 0
    # Variable to store the optimum features
    nof = 0
    score_list = []
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = LinearRegression()
        rfe = RFE(model, nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe, y_test)
        score_list.append(score)
        if (score > high_score):
            high_score = score
            nof = nof_list[n]
    # print("Optimum number of features: %d" % nof)
    # print("Score with %d features: %f" % (nof, high_score))
    cols = list(X.columns)
    model = LinearRegression()
    # Initializing RFE model
    rfe = RFE(model, nof)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X, y)
    # Fitting the data to model
    model.fit(X_rfe, y)
    temp = pd.Series(rfe.support_, index=cols)
    selected_features_rfe = temp[temp == True].index

    quality_measure = nof/len(features)
    return quality_measure

def coverage_gap(df, interval):
    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types
    count = 0
    for key in predicted:
        if predicted[key] in ['datetime64[ns]', 'date-eu', 'date-iso-8601', 'date-non-std-subtype', 'date-non-std']:
            count += 1
            df[key] = pd.to_datetime(df[key])
            new_df = df.resample(interval, on=key, base=0).mean()
            missing = new_df.isna().count()
            quality_measure = (len(new_df) - missing) / len(new_df) * 100

    if count == 0:
        quality_measure = 1
    return quality_measure


def quality_band_A(df, target, interval):
    '''
    Performs all quality measures of band B in one function.
    Parameters
    ----------
    df : Dataframe that needs to be checked.
    file_path : path to the dataframe that needs to be checked.

    Returns
    -------
    out_df : Quality measures of band B in a DataFrame format.
    '''
    inter = interpretable_values(df)
    scaling = feature_scaling(df)
    outlier = outlier_detection(df)
    selection = feature_selection(df, target)
    coverage = coverage_gap(df, interval)

    output_lst = [inter, scaling, outlier, selection, coverage]
    index = ['interpretable values', 'scaling', 'outlier', 'feature selection', 'gap-coverage']

    out_df = pd.DataFrame(output_lst, index=index, columns=['Measures'])
    return out_df



# path = "C:/DataScience/ptype-datasets/main/main/data.gov/3397_1"
# df = pd.read_csv(path + '/data.csv')
# a = feature_selection(df)
# print(a)

# path = "C:/Users/20175848/Dropbox/Data Science Y3/Cognitive science"
# df = pd.read_csv(path + '/rec_tracks.csv')
# a = quality_band_C(df, 'timestamp', '3Y')
# print(a)

# path = "C:/Users/20175848/Dropbox/Data Science Y2/Q4/Business analytics/R/Business analytics/HPI.csv"
# df = pd.read_csv(path, sep=';')
# a = quality_band_C(df, 'timestamp', '3Y')
# print(a)