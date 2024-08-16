import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint

# Load dataset
data = pd.read_csv(r"C:/Users/Mrlaptop/Downloads/codexcue internship projects/Hyperparameter tuning/emails.csv")

# Print column names and types
print("Columns in dataset:")
print(data.columns)
print("Data types:")
print(data.dtypes)

# Separate features and target
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical columns:")
print(categorical_columns)

# Define a transformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# Transform the data
X_transformed = preprocessor.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Define the model
model = RandomForestClassifier()

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and best score for Grid Search
print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Score:", grid_search.best_score_)

# Evaluate the Grid Search model
y_pred_grid = grid_search.best_estimator_.predict(X_test)
print("Grid Search Accuracy:", accuracy_score(y_test, y_pred_grid))
print("Grid Search Classification Report:\n", classification_report(y_test, y_pred_grid))

# Random Search
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters and best score for Random Search
print("Random Search Best Parameters:", random_search.best_params_)
print("Random Search Best Score:", random_search.best_score_)

# Evaluate the Random Search model
y_pred_random = random_search.best_estimator_.predict(X_test)
print("Random Search Accuracy:", accuracy_score(y_test, y_pred_random))
print("Random Search Classification Report:\n", classification_report(y_test, y_pred_random))
