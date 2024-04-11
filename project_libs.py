from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import project_libs as libs
import pandas as pd
from scipy import stats
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
#----------------------------------------------------------------------------------------------------------------------

def read_csv_to_dataframe(file_path):
    """
    Read a CSV file into a pandas DataFrame.
    Parameters:
    - file_path (str): Path to the CSV file to be read.
    Returns:
    - df (pandas.DataFrame): DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f'shape {df.shape}')
        print('-'*100)
        print('List of columns')
        print(df.columns.to_list())
        print('-'*100)
        print('Data info')
        print(df.info())
        print('-'* 100)
        return df
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return None
    except Exception as e:
        print(f"Error occurred while reading the CSV file: {str(e)}")
        return None
#-------------------------------------------------------------------------------------------------------------------------------
def preprocess_missing_values(df):
    """
    Preprocess a DataFrame to handle missing values.
    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data.
    Returns:
    - df (pandas.DataFrame): Preprocessed DataFrame with no missing values.
    """
    try:
        missing_values_sum = df.isnull().sum()
        print(missing_values_sum)
        if missing_values_sum.sum() == 0:
            print("No missing values found. No imputation needed.")
            return df
        else:
            # For numerical columns, fill missing values with the median
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col].fillna(df[col].median(), inplace=True)
            # For categorical columns, fill missing values with a placeholder string 'Unknown'
            for col in df.select_dtypes(include=['object']).columns:
                df[col].fillna('Unknown', inplace=True)
        return df  # Move this line outside the else block
    except NameError as e:
        print(e)
        return None

#---------------------------------------------------------------------------------------------------------------------------------
def split_data(data, target_column, split_size=0.2, random_state=42):
    """
    Splits the data into features (X) and target (y), and further splits them into
    training and testing sets.

    Parameters:
    - data: Pandas DataFrame, input data
    - target_column: str, the name of the target column
    - split_size: float, the proportion of the data to include in the test split (default is 0.2)
    - random_state: int or None, seed for random number generation (default is None)

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing sets for features (X) and target (y)
    
    - # Example usage:
    # Assuming 'your_data' is your DataFrame and 'your_target_column' is the target column name
    X_train, X_test, y_train, y_test = split_data(your_data, 'your_target_column', split_size=0.2, random_state=42)
    """
    try:
        print(f'Data Frame shape {data.shape}')
        # Extract features (X) and target (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Perform the initial split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=random_state)
        print(f'{X_train.shape, X_test.shape, y_train.shape, y_test.shape}')
        return X_train, X_test, y_train, y_test
    except ValueError as ve:
        print(f"Error: {ve}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None
#----------------------------------------------------------------------------------------------------------------------------------------

def Feature_Encoding(df, features=None, encoding_method='one_hot'):
    """
    Apply feature encoding (one-hot encoding or label encoding) to specified features in a DataFrame.

    Parameters:
    - df: DataFrame containing the data
    - features: List of feature column names to encode
    - encoding_method: Method for encoding categorical features ('one_hot' or 'label')

    Returns:
    - DataFrame with encoded features
    """
    try:
        # Check if df is provided and is a DataFrame
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Please provide a valid DataFrame.")
        
        # Check if features list is provided
        if features is None or not isinstance(features, list) or len(features) == 0:
            raise ValueError("List of features must be provided.")

        # Check if encoding method is valid
        if encoding_method not in ['one_hot', 'label']:
            raise ValueError("Invalid encoding method. Please choose 'one_hot' or 'label'.")

        if encoding_method == 'one_hot':
            encoder = OneHotEncoder()
            for col in features:
                # Fit and transform on training data
                # encoder.fit(x_train[col])
                df[col] = encoder.fit_transform(df[col])
                # Transform on test data
                df[col] = encoder.fit_transform(df[col])
        else:
            encoder = LabelEncoder()
            for col in features:
                # encoder.fit(x_train[col])
                df[col] = encoder.fit_transform(df[col])
                df[col] = encoder.fit_transform(df[col])

        return df

    except Exception as e:
        print(f"An error occurred: {str(e)}")

#----------------------------------------------------------------------------------------------------------
def plot_correlation(data):
    """
    Plot correlation matrix using Seaborn.

    Parameters:
    - data: Pandas DataFrame, the input data for which to calculate and visualize the correlation.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap using Seaborn
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    # Show the plot
    plt.title('Correlation Matrix')
    plt.show()

#-----------------------------------------------------------------------------------------------------------------

def create_pair_plot(data):
    """
    Create a pair plot of all columns in the given DataFrame.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame.

    Returns:
    - None
        Displays the pair plot.
    """
    sns.set(style="ticks")
    sns.pairplot(data)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------
def run_ml_pipeline(df, target_column, features, models, encoding_method='label', use_cross_validation=True):
    """
    Run a pipeline of machine learning models on preprocessed data.

    Parameters:
    - df: DataFrame containing the data
    - target_column: Name of the target column
    - features: List of feature column names
    - models: List of tuples (model_name, model_instance, model_parameters)
    - encoding_method: Method for encoding categorical features
    - use_cross_validation: Whether to use cross-validation or model scoring
    """
    try:
        # Check if df is provided
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Please provide a valid DataFrame.")
        
                # Check if features list is provided
        if features is None or not isinstance(features, list) or len(features) == 0:
            raise ValueError("Please provide a valid list of feature column names.")

        # Check if target column is provided
        if target_column is None or not isinstance(target_column, str) or target_column not in df.columns:
            raise ValueError("Please provide a valid target column name that exists in the DataFrame.")

        # Check if models list is provided
        if models is None or not isinstance(models, list) or len(models) == 0:
            raise ValueError("Please provide a valid list of models.")
        
        # Validate the format of each tuple in the models list
        for model_tuple in models:
            if not isinstance(model_tuple, tuple) or len(model_tuple) != 3:
                raise ValueError("Each model tuple must have three elements: (model_name, model_instance, model_parameters).")
            model_name, model_instance, model_params = model_tuple
            if not isinstance(model_name, str) or not hasattr(model_instance, '__call__') or not isinstance(model_params, dict):
                raise ValueError("Invalid format for model tuple. Example format: ('Random Forest', RandomForestRegressor, {'n_estimators': 100}).")

        # Preprocessing
        print('_'*100)
        df = preprocess_missing_values(df)
        print('_'*100)
        df = Feature_Encoding(df=df, features=features, encoding_method=encoding_method)
        print('_'*100)
        df = remove_outliers_zscore(df)
        X_train, X_test, y_train, y_test = split_data(df, target_column=target_column, split_size=0.2, random_state=42)
        print('_'*100)
        x_scaler = StandardScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(x_scaler.transform(X_train), columns = X_train.columns)
        X_test_scaled = pd.DataFrame(x_scaler.transform(X_test),columns = X_train.columns)
        # Model evaluation
        score_test = []
        score_training = []
        model_names = []

        for model_name, model, params in models:
            # Create a pipeline for each model
            pipeline = Pipeline([
                ('model', model(**params))
            ])

            if use_cross_validation:
                # Evaluate the model using cross-validation
                scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5)
                mean_score = scores.mean()
                std_score = scores.std()
                print(f"Model: {model_name}")
                print(f"Mean Accuracy: {mean_score:.2f}")
                print(f"Standard Deviation: {std_score}")
                print("-" * 40)
                score_training.append(mean_score)
                score_test.append(np.nan)  # Cross-validation doesn't provide test scores
            else:
                # Fit the model and compute scores on training and test sets
                train_score = pipeline.fit(X_train_scaled, y_train).score(X_train_scaled, y_train)
                test_score = pipeline.score(X_test_scaled, y_test)
                print(f"Model: {model_name}")
                print(f"Training Score: {train_score:.2f}")
                print(f"Test Score: {test_score:.2f}")
                print("-" * 40)
                score_training.append(train_score)
                score_test.append(test_score)

            model_names.append(model_name)

        result = pd.DataFrame({'Model Name': model_names, 'Training Score': score_training, 'Test Score': score_test})
        return X_train_scaled, X_test_scaled, y_train, y_test, result

    except Exception as e:
        print(f"An error occurred: {str(e)}")

#--------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model, param_grid, X_train, y_train, scoring):
    """
    Perform hyperparameter tuning using Randomized Search CV on a given model.

    Parameters:
    - model: Estimator object (model to be tuned)
    - param_grid: Dictionary of hyperparameter distributions to search
    - X_train: Training features
    - y_train: Training target variable
    - scoring: Scoring metric to optimize during tuning

    Returns:
    - best_model: Best estimator model after tuning
    - best_params: Best parameters found during tuning
    - best_score: Best score achieved during tuning
    """
    # Perform Randomized Search CV
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring, n_iter=100, cv=5, random_state=42)
    random_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_

    return best_model, best_params, best_score


# Custom function to remove scientific notation
def format_price(price):
    return '{:.2f}'.format(price)


def remove_outliers(df, threshold=1.5):
    """
    Remove outliers from all columns of a DataFrame using the IQR method.

    Parameters:
    - df: DataFrame
    - threshold: float, optional (default=1.5)
        Threshold value for identifying outliers. Data points beyond this threshold
        multiplied by the IQR will be considered outliers.

    Returns:
    - DataFrame without outliers
    """
    # Copy the original DataFrame to avoid modifying the input DataFrame
    df_cleaned = df.copy()

    # Iterate through each column
    for column in df.columns:
        # Calculate the IQR for the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds to identify outliers
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Remove outliers based on the bounds
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

    return df_cleaned

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def remove_outliers_zscore(df, z_threshold=3):
    """
    Remove outliers from all columns of a DataFrame using the Z-score method.

    Parameters:
    - df: DataFrame
    - z_threshold: float, optional (default=3)
        Threshold value for identifying outliers. Data points with Z-scores beyond
        this threshold will be considered outliers.

    Returns:
    - DataFrame without outliers
    """
    # Copy the original DataFrame to avoid modifying the input DataFrame
    df_cleaned = df.copy()

    # Iterate through each column
    for column in df.columns:
        # Calculate the Z-scores for each data point in the column
        z_scores = stats.zscore(df[column])

        # Identify and remove outliers based on the Z-scores
        outliers_mask = (abs(z_scores) <= z_threshold)
        df_cleaned = df_cleaned[outliers_mask]

    return df_cleaned