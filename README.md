# Analyzing Weather Impacts on Crop Production in California Counties

## Overview
This project explores the relationship between extreme weather conditions and crop performance across California counties. By integrating historical weather and crop yield data, the analysis identifies key weather variables influencing crop outcomes. Additionally, the project develops predictive models and visualizations to assist agricultural stakeholders in mitigating risks and optimizing crop management.

---

## Goals

1. **Assess Weather-Crop Relationships**:
   - Analyze the impact of extreme weather variables (e.g., high temperature days, heavy rainfall) on crop outcomes.
   - Identify county-specific weather variables that significantly affect yields for key crops like almonds, tomatoes, and wine grapes.

2. **Predict Crop Yields**:
   - Develop predictive models using machine learning (e.g., Random Forest, Ridge Regression) to forecast crop outcomes.
   - Incorporate lagged weather and crop variables to enhance model accuracy.

3. **Empower Decision-Making**:
   - Provide actionable insights and interactive visualizations to support farmers and agricultural planners.

---

## Features

### **Data Analysis and Modeling**
- **Weather Feature Engineering**:
  - Calculated extreme weather metrics (e.g., `high_temp_days`, `heavy_rain_days`) for each county using location-specific thresholds.
- **Statistical Analysis**:
  - Regression analysis and hypothesis testing to identify significant predictors for crop outcomes.
- **Machine Learning Models**:
  - Built models to predict yield and production using weather and lagged crop variables.

### **Visualizations**
- **Interactive Dashboards**:
  - Simulate extreme weather impacts using dynamic plots.
  - Explore crop yield patterns and farming practices through geographical heatmaps.
- **Static Insights**:
  - Compare historical yield data under varying weather conditions using bar plots and scatter plots.

---

## Dataset Highlights

- **Weather Data**: Daily and yearly metrics sourced from NOAA and OpenWeatherMap API.
- **Crop Data**: USDA crop yield data spanning 1980 to present.
- **Key Crops**:
  - **Almonds** (Fresno, Tulare, Kings)
  - **Tomatoes** (Fresno, Kings, Santa Clara)
  - **Wine Grapes** (Napa, Sonoma, Mendocino)

---

## Key Findings

1. **Significant Weather Predictors**:
   - **High Wind Days** negatively affect grape yields in Alameda and Mendocino.
   - **High Temperature Days** benefit almond yields in Tulare but hinder tomato production in Fresno.
   
2. **Predictive Modeling**:
   - Lagged features improved model accuracy but highlighted variability across counties and crops.
   - Simpler regression models often outperformed machine learning models on smaller datasets.

3. **Visual Insights**:
   - Correlation heatmaps and time series plots revealed critical weather-crop relationships and regional variations.

---

## How to Use

### Clone the Repository
```bash
git clone https://github.com/your-username/weather-crop-analysis.git
```

### Run the Application
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Dash app:
   ```bash
   python app.py
   ```
3. Access the dashboard at `http://127.0.0.1:8050`.

#### Colab Notebook (Analysis and ML Model): https://colab.research.google.com/drive/1xZXZ7lU_aWVcM8PXUi_K44Iom1WTDiGp?usp=sharing
#### Colab Notebook (Visualization Plan): https://colab.research.google.com/drive/1eXGaCdHmqq_IcxItqU5Y1ujAiClHqDID?usp=sharing
