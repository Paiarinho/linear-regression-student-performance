import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_test, y_pred, report=True, title='Model Evaluation'):

    evaluation = {
        'mae' : mean_absolute_error(y_test, y_pred),
        'mse' : mean_squared_error(y_test, y_pred),
        'r2' : r2_score(y_test, y_pred)
    }
    rmse = np.sqrt(evaluation['mse'])
    evaluation['rmse'] = rmse

    if report:
        print(title)
        print(f"R² (R-squared): {evaluation['r2']:.4f}")
        print(f"MAE (Mean Absolute Error): {evaluation['mae']:.2f}")
        print(f"RMSE (Root Mean Square Error): {evaluation['rmse']:.2f}")

    return evaluation

def plot_residual_analysis(y_test, y_pred):
    """
    Plots residual analysis for regression models including:
    1. Residuals vs Predicted Values
    2. Histogram of Residuals
    3. Q-Q Plot
    """
    residuals = y_test - y_pred

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Residuals vs Predicted Values (Homoscedasticity and Linearity)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax[0])
    ax[0].axhline(y=0, color='r', linestyle='--')
    ax[0].set_title('Residuals vs Predicted Values')
    ax[0].set_xlabel('Predicted Values')
    ax[0].set_ylabel('Residuals')

    # 2. Histogram of Residuals (Normality)
    sns.histplot(residuals, kde=True, bins=30, ax=ax[1])
    ax[1].set_title('Residuals Distribution')
    ax[1].set_xlabel('Residuals')

    # 3. Q-Q Plot (Normality)
    stats.probplot(residuals, dist="norm", plot=ax[2])
    ax[2].set_title('Q-Q Plot of Residuals')

    plt.tight_layout()
    plt.show()