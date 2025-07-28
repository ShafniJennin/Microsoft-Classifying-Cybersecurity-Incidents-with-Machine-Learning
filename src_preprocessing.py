import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess(df):
    df = df.copy()

    # Drop columns with too many missing values (optional)
    null_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < null_threshold]

    # Fill missing values
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='median')

    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Encode categorical variables
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df
