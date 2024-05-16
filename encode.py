from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def encode_data(df):
    ct = ColumnTransformer(transformers=[
        ('ohe',OneHotEncoder(sparse_output=False),['Extracurricular Activities'])
    ],remainder='passthrough')

    ct_trans = ct.fit_transform(df)

    ct_data = pd.DataFrame(ct_trans,columns=ct.get_feature_names_out())

    return ct_data


