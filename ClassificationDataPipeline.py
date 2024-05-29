import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
        "KNN": KNeighborsRegressor(),
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

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        accuracy_scores["train"][model_name] = train_accuracy
        accuracy_scores["test"][model_name] = test_accuracy
        print(f'{model_name} - Training Score: {train_accuracy:.4f}')
        print(f'{model_name} - Testing Score: {test_accuracy:.4f}')
        print('-' * 50)

    print('Final Accuracy Scores:')
    for key in accuracy_scores:
        print(f"{key.capitalize()} Scores:")
        for model_name, score in accuracy_scores[key].items():
            print(f"  {model_name}: {score:.4f}")
        print()

    return accuracy_scores



# Example usage
# df = pd.read_csv('your_data.csv')
# target_column = 'your_target_column'
# Randomstate = 42
# accuracy_scores = train_and_evaluate_models(df, target_column, Randomstate)




# df = pd.read_csv('wines_SPA.csv')
# target_column = 'rating'
# Randomstate = 50
# accuracy_scores = train_and_evaluate_models(df, target_column, Randomstate)
# print(accuracy_scores)
