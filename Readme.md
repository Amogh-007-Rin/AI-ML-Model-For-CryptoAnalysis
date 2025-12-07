# 📈 Cryptocurrency Market Direction Prediction using Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Abstract

This project investigates the efficacy of **Sentiment Analysis** and **Machine Learning** in predicting the short-term (24-hour) price direction of major cryptocurrencies. Challenging the Efficient Market Hypothesis (EMH), this study explores whether social media sentiment, news scores, and technical indicators contain predictive signals that can be exploited by non-linear algorithms.

This work was completed as part of the **CMP5367: Artificial Intelligence and Machine Learning** module.

## 📊 Dataset

The project utilizes the **Crypto Market Sentiment and Price Dataset (2025)** sourced from Kaggle.
* **Source:** [Kaggle Link](https://www.kaggle.com/datasets/pratyushpuri/crypto-market-sentiment-and-price-dataset-2025)
* **Observations:** 2,063 data points.
* **Features:** Social Sentiment Score, News Impact Score, Fear & Greed Index, RSI, Volatility, Market Cap, and Volume.
* **Target:** Binary Classification (1 = Bullish/Up, 0 = Bearish/Down).

## 🛠️ Methodology

### 1. Data Pre-processing
* **Target Engineering:** Converted continuous `% Price Change` into a binary target.
* **Cleaning:** Verified data integrity and handled potential leakage features.
* **Transformation:**
    * **One-Hot Encoding** for categorical assets (e.g., Bitcoin, Ethereum).
    * **Standard Scaling** for numerical features to normalize disparate ranges (e.g., Volume vs. Sentiment Score).

### 2. Models Evaluated
Four distinct algorithm families were tested to compare linear vs. non-linear performance:
1.  **Logistic Regression** (Linear Baseline)
2.  **Random Forest Classifier** (Bagging Ensemble)
3.  **XGBoost (Gradient Boosting)** (Boosting Ensemble)
4.  **Support Vector Machine (SVM)** (Kernel-based)

### 3. Optimization
To ensure rigorous evaluation, **GridSearchCV** (3-Fold Cross-Validation) was applied to **all** candidate models to optimize hyperparameters (e.g., Tree Depth, Learning Rate, C-value).

## 🏆 Results

The experiment revealed that market sentiment has a very low linear correlation with price. However, non-linear ensemble methods successfully extracted a marginal predictive edge over random guessing.

| Model | Baseline Accuracy | Tuned Accuracy | Improvement |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 49.1% | **52.1%** 🏆 | **+3.0%** |
| **XGBoost** | 46.5% | **51.8%** | +5.3% |
| **SVM** | 45.8% | **50.6%** | +4.8% |
| **Logistic Regression** | 49.6% | **50.1%** | +0.5% |

**Key Finding:** The **Random Forest** model with `max_depth=10` achieved the highest stability and accuracy, suggesting that regularization is key when modeling high-noise financial data.

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* Pip

### Setup
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Amogh-007-Rin/AI-ML-Model-For-CryptoAnalysis](https://github.com/Amogh-007-Rin/AI-ML-Model-For-CryptoAnalysis)
    cd AI-ML-Model-For-CryptoAnalysis
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis**
    You can run the full analysis pipeline using the provided Python script or Jupyter Notebook:
    ```bash
    python main.py
    ```
    *Or launch Jupyter:*
    ```bash
    jupyter notebook notebooks/Crypto_Trend_Analysis.ipynb
    ```

## 📂 Project Structure

```text
└── Academic_Reports/       # Final Academic Report (D1.b).
├── dataset/                # Dataset files (csv).
├── Final_Notebook/         # Jupyter Notebooks containing complete project and ML model code [ Main execution script ] with a markdown file.
├── Ideas/                  # Model training ideas and project discussions proposed by group members.
├── images/                 # Plots and correlation heatmaps.               
├── Project-Brief/          # Contains the project briefing and sample reports.
├── src/                    # Source code for different sections of the project - [Dataset Overview, EDA, Dataset Pre-Processing, Model-Training And Evaluation].
├── README.md               # Project documentation.
├── requirements.txt        # Python dependencies.