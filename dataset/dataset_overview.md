# Dataset Overview

Your dataset contains **2063 rows** and **14 columns**, focused on
cryptocurrency market data, technical indicators, and sentiment metrics
used for **price‑movement prediction**.

------------------------------------------------------------------------

## 📊 Column Details

  ------------------------------------------------------------------------------
  Column Name                    Description               Data Type
  ------------------------------ ------------------------- ---------------------
  **timestamp**                  Date/time of the record   object

  **cryptocurrency**             Name of the               object
                                 cryptocurrency            

  **current_price_usd**          Current market price      float
                                 (USD)                     

  **price_change_24h_percent**   \% price change in the    float
                                 last 24h                  

  **trading_volume_24h**         Total trading volume in   float
                                 the last 24h              

  **market_cap_usd**             Market capitalization     float
                                 (USD)                     

  **social_sentiment_score**     Sentiment score from      float
                                 social media              

  **news_sentiment_score**       Sentiment from news       float
                                 sources                   

  **news_impact_score**          Strength/impact of news   float
                                 events                    

  **social_mentions_count**      Number of social media    float
                                 mentions                  

  **fear_greed_index**           Market emotion indicator  float
                                 (0--100)                  

  **volatility_index**           Market volatility measure float

  **rsi_technical_indicator**    RSI value                 float

  **prediction_confidence**      Confidence score of       float
                                 prediction                
  ------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📉 Missing Values

No missing values were found in the dataset.

------------------------------------------------------------------------

## 📈 Statistical Summary Highlights

### **Price & Market Features**

-   **current_price_usd**: mean \~22,262 (min: 0.94, max: 68,012)\
-   **market_cap_usd**: mean \~279B (max: 1.29T)\
-   **trading_volume_24h**: mean \~11.2B (max: 95B)

### **Sentiment Features**

-   **social_sentiment_score**: avg \~0.48\
-   **news_sentiment_score**: avg \~0.23\
-   **news_impact_score**: avg \~0.50

### **Technical Indicators**

-   **RSI**: mean \~49.3 (range: 1.4--97.1)\
-   **Volatility Index**: avg \~12.46

### **Prediction Feature**

-   **prediction_confidence**: mean = 77.12 (range: 55.9--100)

------------------------------------------------------------------------

## 📌 Notes

This dataset is well‑structured, clean, and suitable for predictive
modeling, sentiment analysis, and time‑series analytics.
