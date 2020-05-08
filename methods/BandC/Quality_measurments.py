import pandas as pd
from Pywash2.methods.BandC.ParserUtil import assign_parser

def parse_ability_measure(df):
    '''Checks whether the application read the data appropriately in terms of parsing

    Parameters
    ----------

    df : DataFrame
        DataFrame containing the data.

    Returns
    -------
    quality_measure : The quality measure of parse-ability as an integer
    '''

    if not df.empty:
        quality_measure = 1
        print('Quality measurement for parse-ablity = 1')
    else:
        quality_measure = 0
        print('Quality measurement for parse-ablity = 0')
    return quality_measure

def data_storage_measure(df):
    '''
    Checks whether the application can perform algorithms on given data.

    Parameters
    ----------

    df : DataFrame containing the data.

    Returns
    -------
    quality_measure : The quality measure of storage as an integer
    '''
    try:
        ##TODO
        heftigalgoritme()
        quality_measure = 1
    except:
        print('The volume of the data is too big for our algorithms.')
        quality_measure = 0
    return quality_measure

def encoding_measure(file_path):
    '''
    Checks whether the automated encoding detection works on the given dataset.

    Parameters
    ----------

    File path : Path of the data set location.

    Returns
    -------
    quality_measure : Quality measure of encoding, either known encoding, or not.
    '''
    test = assign_parser(file_path)

    if test == None:
        quality_measure = 0
    else:
        quality_measure = 1
    return(quality_measure)

def data_formats_measure(df):
    '''
    Checks whether the data formatting is done appropriately and calculates a score.

    Parameters
    ----------

    df : Dataframe containing the data

    Returns
    -------
    quality_measure : Quality measure in terms of data formatting in the form of an integer.
    '''
    thomas_zn_algoritme()

    quality_measure = certainty * (number_expected / total_number)
    return quality_measure

def disjoint_datasets_measure():
    '''
    Checks whether the dataset is dis-joint. (never the case with PyWash, so measure = 1)

    Parameters
    ----------

    Returns
    -------
    quality_measure : Quality measure in terms of data formatting in the form of an integer.
    '''
    quality_measure = 1
    return quality_measure

def quality_band_C(df, file_path):
    '''
    Performs all quality measures of band C in one function.
    Parameters
    ----------
    df : Dataframe that needs to be checked.
    file_path : path to the dataframe that needs to be checked.
    Returns
    -------
    quality_measure : Quality measure of band C.
    '''
    parse = parse_ability_measure(df)
    storage = data_storage_measure(df)
    encoding = encoding_measure(file_path)
    format = data_formats_measure(df)
    disjoint = disjoint_datasets_measure()

    print('The quality of parsing = {}.').format(parse)
    print('The quality of storage = {}.').format(storage)
    print('The quality of encoding = {}.').format(encoding)
    print('The quality of formatting = {}.').format(format)
    print('The quality of disjoint datasets = {}.').format(disjoint)

    return (parse + storage + encoding + format + disjoint)
