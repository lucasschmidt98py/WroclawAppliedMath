import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV , KFold
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data():
    """
    Load the dataset and preprocess features.
    Returns:
    - X_train, X_test, y_train, y_test, feature_names
    """
    # Load the dataset
    housing = fetch_openml(name="house_prices", as_frame=True)
    X = housing.data
    y = housing.target

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    numeric_feature_names = numeric_cols.tolist()
    categorical_feature_names = (
        preprocessor.named_transformers_['cat']['onehot']
        .get_feature_names_out(categorical_cols)
    )
    feature_names = numeric_feature_names + categorical_feature_names.tolist()
    X_processed = X_processed.toarray()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def visualize_data(X, y , features):
    """
    Create scatter plots to visualize relationships between features and target.
    Your code should:
    1. Create scatter plots for each feature vs. house price
    2. Add trend lines
    3. Add proper labels and titles
    """
    nfeatures = len(features)
    ncols = 3
    nrows = math.ceil( nfeatures / ncols )
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    for i in range(len(features)):
        ax = axes[i // ncols , i % ncols]
        sns.scatterplot(x=X[features[i]], y=y, ax=ax)
        ax.set_ylabel("House Price")
        ax.set_title(f"{features[i]} vs. House Price")
    plt.tight_layout()
    plt.show()

def train_linear_models(X_train, y_train):
    """
    Train different types of linear regression models.

    Implement:
    1. Simple Linear Regression
    2. Ridge Regression (L2 regularization)
    3. Lasso Regression (L1 regularization)
    4. Elastic Net Regression

    Returns:
    - Dictionary containing trained models
    """
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def perform_cross_validation(model, X, y, k=5):
    """
    Perform k-fold cross-validation for regression models.

    Your code should:
    1. Implement k-fold cross-validation
    2. Calculate MSE and R² for each fold
    3. Return mean and std of performance metrics

    Parameters:
    - model: The regression model to evaluate
    - X: Feature matrix
    - y: Target vector
    - k: Number of folds

    Returns:
    - Dictionary with cross-validation metrics
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mse_scores = -mse_scores
    results = {
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores)
    }
    return results

def tune_model_parameters(X_train, y_train):
    """
    Perform grid search for hyperparameter tuning.

    Parameters:
    - X_train: Training data features
    - y_train: Training data target values

    Returns:
    - Dictionary with best parameters for each model
    """
    param_grids = {
        'ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'elastic_net': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }

    models = {
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elastic_net': ElasticNet()
    }

    best_params = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[model_name] =  grid_search.best_params_
    return best_params

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models using multiple metrics.

    Parameters:
    - models: Dictionary of trained models
    - X_test: Feature matrix for the test set
    - y_test: Actual target values for the test set

    Returns:
    - Dictionary with evaluation metrics for each model
    """
    evaluation_metrics = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluation_metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    return evaluation_metrics


def visualize_coefficients(models, features):
    df = pd.DataFrame({'features':features})
    for name , model in models.items():
        df[name] = model.coef_
    df = df.melt(id_vars='features', var_name='Model', value_name='Coefficient')
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df, x='features', y='Coefficient', hue='Model')
    plt.show()

def select_features():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    all_features = X_train.columns.tolist()
    best_params = tune_model_parameters(X_train,y_train)
    final_models = {
    'ridge': Ridge(**best_params['ridge']).fit(X_train , y_train),
    'lasso': Lasso(**best_params['lasso']).fit(X_train , y_train),
    'elastic_net': ElasticNet(**best_params['elastic_net']).fit(X_train , y_train)
    }
    df = pd.DataFrame({} , index=all_features)
    for name , model in final_models.items():
        df[name] = model.coef_
    return df.loc[ df.lasso != 0 ]
