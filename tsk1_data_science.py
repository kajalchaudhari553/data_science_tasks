import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # for saving pipeline

# 1. Extract: Load data from CSV
def extract_data(url):
    print("Extracting data...")
    df = pd.read_csv(url)
    return df

# 2. Transform: Clean and prepare the dataset
def transform_data(df):
    print("Transforming data...")

    # Drop irrelevant columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Define numeric and categorical features
    num_features = ['Age', 'Fare']
    cat_features = ['Sex', 'Embarked', 'Pclass']

    # Imputation + Scaling for numeric features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Imputation + One-hot encoding for categorical features
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both pipelines
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Fit and transform features
    X_transformed = full_pipeline.fit_transform(X)

    return X_transformed, y, full_pipeline

# 3. Load: Split and save the processed data
def load_data(X, y):
    print("Loading data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 4. Run full ETL pipeline
def run_etl():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = extract_data(url)
    X, y, pipeline = transform_data(df)
    X_train, X_test, y_train, y_test = load_data(X, y)

    # Save the pipeline
    joblib.dump(pipeline, 'titanic_preprocessing_pipeline.pkl')
    print("ETL completed and pipeline saved.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_etl()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

