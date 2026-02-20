# ============================================================
# HOUSING MARKET TRENDS ANALYSIS
# Visualizing Sale Prices and Property Features
# ============================================================

# ========================
# 1. IMPORT LIBRARIES
# ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Optional: comment out if not installed
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")


# ========================
# 2. LOAD DATA
# ========================

def load_data(path):
    df = pd.read_csv(path)
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    return df


# ========================
# 3. CLEAN DATA
# ========================

def clean_data(df):
    df = df.dropna()
    df = df[df['price'] > 0]
    df = df[df['sqft'] > 0]
    return df


# ========================
# 4. FEATURE ENGINEERING
# ========================

def create_features(df):
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    
    df = df.sort_values('sale_date')
    
    # Year-over-Year Growth (12 periods)
    df['yoy_growth'] = df['price'].pct_change(periods=12)
    
    return df


# ========================
# 5. KPIs
# ========================

def calculate_kpis(df):
    print("\n========== KEY PERFORMANCE INDICATORS ==========")
    print("Median Sale Price:", round(df['price'].median(), 2))
    print("Average Price per Sqft:", round(df['price_per_sqft'].mean(), 2))
    print("Total Transactions:", len(df))
    print("Average YoY Growth:", round(df['yoy_growth'].mean(), 4))


# ========================
# 6. VISUALIZATIONS
# ========================

def plot_price_trend(df):
    trend = df.groupby('sale_date')['price'].median()

    plt.figure(figsize=(12,6))
    trend.plot()
    plt.title("Median Sale Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Median Price")
    plt.show()


def plot_price_vs_sqft(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='sqft', y='price')
    plt.title("Price vs Square Footage")
    plt.show()


def plot_correlation_heatmap(df):
    corr = df[['price','sqft','bedrooms','bathrooms','lot_size']].corr()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()


# ========================
# 7. MODELING
# ========================

def train_model(df):
    X = df[['sqft','bedrooms','bathrooms','lot_size']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n========== MODEL PERFORMANCE ==========")
    print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
    print("R2 Score:", round(r2_score(y_test, predictions), 4))

    # Feature Importance (coefficients)
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    print("\nFeature Importance:")
    print(importance)

    return model


# ========================
# 8. MAIN EXECUTION
# ========================

def main():

    # Update path to your dataset
    file_path = "housing_sales.csv"

    # Pipeline
    df = load_data(file_path)
    df = clean_data(df)
    df = create_features(df)

    # KPIs
    calculate_kpis(df)

    # Visualizations
    plot_price_trend(df)
    plot_price_vs_sqft(df)
    plot_correlation_heatmap(df)

    # Model
    model = train_model(df)

    print("\nAnalysis Complete ✅")


# ========================
# RUN SCRIPT
# ========================

if __name__ == "__main__":
    main()
