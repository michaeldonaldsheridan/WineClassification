import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate_models(df: pd.DataFrame, target_column: str, randomstate: int):
    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    models = {
        "KNN": KNeighborsRegressor(n_neighbors=2),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "SVM": SVR(),
        "Linear Regression": LinearRegression()
    }

    pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                 for name, model in models.items()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randomstate)

    accuracy_scores = {
        "train": {},
        "test": {}
    }

    r2_scores = {
        "train": {},
        "test": {}
    }

    mse_scores = {
        "train": {},
        "test": {}
    }

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        accuracy_scores["train"][model_name] = train_score
        accuracy_scores["test"][model_name] = test_score
        
        r2_scores["train"][model_name] = train_r2
        r2_scores["test"][model_name] = test_r2

        mse_scores["train"][model_name] = train_mse
        mse_scores["test"][model_name] = test_mse
        
        print(f'{model_name} - Training Score: {train_score:.4f}, R² Score: {train_r2:.4f}, MSE: {train_mse:.4f}')
        print(f'{model_name} - Testing Score: {test_score:.4f}, R² Score: {test_r2:.4f}, MSE: {test_mse:.4f}')
        print('-' * 50)

        
    print('Final Accuracy, R², and MSE Scores:')
    for key in accuracy_scores:
        print(f"{key.capitalize()} Scores:")
        for model_name, score in accuracy_scores[key].items():
            print(f"  {model_name}: Accuracy Score: {score:.4f}, R² Score: {r2_scores[key][model_name]:.4f}, MSE: {mse_scores[key][model_name]:.4f}")
        print()
    


    return accuracy_scores, r2_scores, mse_scores

# Example usage
# df = pd.read_csv('your_data.csv')
# target_column = 'your_target_column'
# randomstate = 42
# accuracy_scores, r2_scores, mse_scores = train_and_evaluate_models(df, target_column, randomstate)

#df = pd.read_csv('wines_SPA.csv')
#target_column = 'rating'
#randomstate = 50
#accuracy_scores, r2_scores, mse_scores = train_and_evaluate_models(df, target_column, randomstate)
#print(accuracy_scores)
#print(r2_scores)
#print(mse_scores)

#Example usage
#df = pd.read_csv('your_data.csv')
#target_column = 'your_target_column'
#randomstate = 42
#accuracy_scores, r2_scores, mse_scores = train_and_evaluate_models(df, target_column, randomstate)

df = pd.read_csv('wines_SPA.csv')
target_column = 'rating'
randomstate = 50
accuracy_scores, r2_scores, mse_scores = train_and_evaluate_models(df, target_column, randomstate)
print(accuracy_scores)
print(r2_scores)
print(mse_scores)
