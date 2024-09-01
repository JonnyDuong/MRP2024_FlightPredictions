import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your data
data = pd.read_csv('training.v.1.csv')
classDF = pd.read_csv('class.v.1.csv')

# Prepare features and target
X = data.copy()
y = classDF.copy()

X = X.iloc[:50000, :]
y = y.iloc[:50000, :]

# Normalize the data
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()

if y.size > 0:
    y_normalized = scaler_y.fit_transform(y)
else:
    raise ValueError("The target variable y is empty.")

y_normalized = y_normalized.squeeze()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Define parameter grids
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['linear', 'rbf']
}

param_grid_gbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize models
models = {
    'Random_Forest': (RandomForestRegressor(random_state=42), param_grid_rf),
    'KNN': (KNeighborsRegressor(), param_grid_knn),
    'SVM': (SVR(), param_grid_svm),
    'GBM': (GradientBoostingRegressor(random_state=42), param_grid_gbm),
    'Linear_Regression': (LinearRegression(), {})
}

# Iterate through models, apply GridSearchCV, and evaluate
for name, (model, param_grid) in models.items():
    
    print(f"\nRunning model: {name}")
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    # Extract results from GridSearchCV
    results = pd.DataFrame(grid_search.cv_results_)

    for param in param_grid.keys():
        results[param] = results['params'].apply(lambda x: x[param])
    
    # Calculate RMSE, MAE, and R-squared from the cross-validation results
    results['RMSE'] = (-results['mean_test_score']) ** 0.5
    results['MSE'] = -results['mean_test_score']
    results['R2'] = results['mean_test_score'].apply(lambda x: 1 - x / np.var(y_test))
    results['MAE'] = results.apply(lambda row: mean_absolute_error(y_test, model.set_params(**row['params']).fit(X_train, y_train).predict(X_test)), axis=1)

    # Pivot table for RMSE across hyperparameter combinations
    if name != 'Linear_Regression':  # Linear Regression has no hyperparameter grid
        pivot_table_rmse = results.pivot_table(values='RMSE', 
                                               index=list(param_grid.keys())[:-1],
                                               columns=list(param_grid.keys())[-1], 
                                               aggfunc='mean')
        print("Pivot Table for RMSE:")
        print(pivot_table_rmse)

        # Pivot table for R2 across hyperparameter combinations
        pivot_table_r2 = results.pivot_table(values='R2', 
                                             index=list(param_grid.keys())[:-1],
                                             columns=list(param_grid.keys())[-1], 
                                             aggfunc='mean')
        print("Pivot Table for R2:")
        print(pivot_table_r2)

    # Store the best hyperparameters
    best_params = grid_search.best_params_

    # Denormalize the predictions and actual values
    y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()

    # Calculate errors
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    r2 = r2_score(y_test_denorm, y_pred_denorm)
    mse = mean_squared_error(y_test_denorm, y_pred_denorm)

    print(f"\nBest Parameters: {best_params}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
    
    # Display first 10 predictions
    errors = np.abs(y_pred_denorm - y_test_denorm)
    error_percentages = (y_pred_denorm - y_test_denorm) / (y_test_denorm + 1e-10)
    results_df = pd.DataFrame({
        'Predicted Value (min)': y_pred_denorm,
        'Actual Value (min)': y_test_denorm,
        'Error (min)': errors,
        'Error (%)': error_percentages * 100
    })

    print(f"First 10 Predictions from Best Hyperparameter Run ({name}):")
    print(results_df.head(10))
    
    output_file = f"Model_{name}_results.txt"

    # Open the file in write mode
    with open(output_file, "w") as f:
        f.write(f"Model: {name}\n")
        
        # Save accuracy metrics
        f.write("Accuracy Metrics:\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R2 Score: {r2}\n")
        f.write(f"MSE: {mse}\n\n")
        
        # Save pivot table RMSE if exists
        if name != 'Linear_Regression':
            f.write("Pivot Table for RMSE:\n")
            f.write(pivot_table_rmse.to_string())
            f.write("\n\n")
        
            f.write("Pivot Table for R2:\n")
            f.write(pivot_table_r2.to_string())
            f.write("\n\n")
        
        # Save first 10 predictions
        f.write("First 10 Predictions:\n")
        f.write(results_df.head(10).to_string())
        f.write("\n\n")
        
        f.write("----------------------------------------------------\n\n")
    
    print(f"Results have been saved to {output_file}.")
