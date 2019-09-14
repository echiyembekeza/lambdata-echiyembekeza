"""
lambdata - A collection of Data Science helper functions.
"""

import pandas as pd

def is_null(df):
    pd.DataFrame.isnull(df)
