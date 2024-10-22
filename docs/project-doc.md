# Proposal: Fresno Weather Data Analysis and Forecasting

## **Project Summary: Weather and Crop Data Analysis and Forecasting**

### **Project Goals:**
The primary goal of this project is to develop two independent, yet interrelated, systems: 
1. A forecasting model for short-term weather conditions in **Fresno County**.
2. A predictive model for **tomato and almond yields**. 

The project will apply time series analysis to forecast **daily weather patterns** such as temperature and precipitation, using data from **1979 to the present**, and develop a model to estimate future crop yields. The goal is to evaluate how weather conditions affect crop production, and simulate potential scenarios that can assist farmers in optimizing planting, irrigation, and harvesting schedules.

### **Specific Crops and Timeframe:**
The project will focus on the following crops, selected due to their economic importance in **Fresno** and the availability of historical yield data:
- **Almonds:** A major crop in Fresno, sensitive to temperature and water availability.
- **Tomatoes:** Widely grown and sensitive to temperature fluctuations, with high economic value.

The analysis will cover the period from **1979 to the present**, utilizing both historical crop yield data and weather data, focusing on **Fresno County**, but incorporating relevant insights from other nearby regions when necessary.

### **Key Features:**

1. **Independent Weather Prediction Model:**
   - A **SARIMA model** will predict daily weather conditions, focusing on key variables such as temperature and precipitation from 1979 to the present.
   - The system will forecast **future daily weather conditions** for a specific time period (e.g., the next 30 days or years), using advanced time series techniques to generate accurate short-term weather predictions, allowing farmers to plan agricultural activities accordingly.
   
2. **Crop Yield Prediction Model:**
   - A separate **time series model** will predict **future crop yields**, focusing on key crop metrics such as **Harvested Acres**, **Yield**, and **Production** for almonds and tomatoes.
   - This model will utilize historical crop data combined with relevant weather variables (e.g., lagged temperature and precipitation) to provide insights into future crop productivity.

3. **Integration of Weather and Crop Models:**
   - Although the models for **weather** and **crop yields** are developed independently, the results from the weather prediction model will be used to **simulate different weather scenarios** and observe how they might affect future crop yields.
   - By adjusting key weather factors (e.g., temperature rise, rainfall variations), the project will demonstrate the impact of potential climate changes on crop productivity in Fresno County.

### **Project Outcome and Usefulness:**
- The final system will provide **short-term weather forecasts** and **future crop yield predictions**, offering a tool for **scenario-based planning**.
- Farmers and agricultural planners can use this tool to make informed decisions about when and how to plant, irrigate, and harvest, optimizing their practices based on both weather forecasts and crop yield predictions.
- By understanding the likely effects of different weather conditions on crop outcomes, the project will help **mitigate risks** associated with temperature and precipitation fluctuations, potentially improving overall agricultural productivity.

The analysis will cover the period from **1979 to the present**, utilizing both historical crop yield data and weather data for each growing season in the Central California region, which includes Fresno County, and various other ``['Fresno', 'Sacramento', 'San Joaquin', 'Stanislaus', 'Merced', 'Madera', 'Kings', 'Tulare',
               'Kern', 'Yolo', 'Colusa', 'Sutter', 'Butte', 'Glenn', 'Tehama']``

---

### **User Implementation and Benefits:**

#### **1. Simple Web-Based Dashboard:**
   - **Implementation:** A straightforward web-based dashboard will provide daily weather forecasts and display visualizations (e.g., temperature trends, rainfall predictions) with a focus on crop yield impact.
     - Users can access real-time weather data and track how forecasted conditions might impact yields for **almonds** and **tomatoes**, based on historical correlations.
     - The dashboard will offer simple, easy-to-read graphs and alerts tailored to these two crops.
   - **Benefit:** Farmers will gain a **quick overview** of the predicted weather and its potential impact on specific crop yields, enabling better decision-making for daily agricultural activities.

#### **2. Basic Alerts for Critical Weather Conditions:**
   - **Implementation:** The system will send basic alerts via SMS or email when critical weather conditions (e.g., excessive heat or lack of rainfall) are forecasted.
     - Users can set preferences for receiving alerts based on conditions likely to affect **almonds** or **tomatoes**, such as heat alerts for almonds or drought warnings for tomatoes.
   - **Benefit:** The alerts will help farmers take **timely action** to protect crops from adverse weather conditions, such as adjusting irrigation or delaying planting during extreme heat.

#### **Data Sources:**
The project will utilize the following key data sources:

1. **NOAA Climate Data:** Provides comprehensive historical weather data, including temperature and precipitation records, essential for analyzing long-term climate trends and patterns in Fresno.

2. **OpenWeatherMap API:** Offers real-time weather data, including current temperature and precipitation, crucial for accurate short-term weather forecasts and anomaly detection.

3. **USDA Crop Yield Data:**
   - Data Available: Provides historical crop yield data for **almonds** and **tomatoes** in Fresno County.
   - Timeframe: Crop yield data from 1979 to the present.


Here's the revised section based on the new methods and approaches implemented in your project:

---

## **Expected Major Findings**

In this project, several key findings are anticipated based on the **independent forecasting models** for weather and crop yield data in **Fresno County**. These findings will provide valuable insights into short-term weather patterns, predictions for **tomato** and **almond** yields, and their interactions through scenario testing. Below are the main areas of exploration and the expected outcomes:

#### **1. Identification of Short-Term Weather Patterns and Scenario Analysis of Their Impact on Crop Yield**
   - **Expected Finding:** 
     - Analyze historical weather data using a **SARIMA model** to identify short-term weather patterns, including daily temperature fluctuations and precipitation trends, particularly during critical growing seasons for **tomatoes** and **almonds**.
     - Simulate various weather scenarios (e.g., hotter-than-usual summers, increased rainfall) and assess how these projected weather conditions could affect the future yields of **tomatoes** and **almonds**.
   - **Value:** 
     - This analysis will provide farmers with a deeper understanding of how **specific weather conditions** (e.g., heatwaves, seasonal rainfall) impact the yields of **tomatoes** and **almonds**. By incorporating short-term weather forecasts into scenario analyses, farmers can better optimize their agricultural practices based on potential future weather events.

#### **2. Development of Independent Weather and Crop Yield Forecasting Models**
   - **Expected Finding:** 
     - Develop a **SARIMA model** to provide accurate short-term forecasts of daily weather conditions (temperature and precipitation) for **Fresno County**.
     - Develop a separate **time series model** for predicting future **tomato** and **almond** yields based on historical trends in harvested acres, yield, and production.
     - Use the predicted weather data to explore potential **indirect effects** of weather scenarios on crop yields, offering valuable insights into how climate trends may influence agricultural productivity.
   - **Value:** 
     - Farmers will be equipped with two complementary tools: **short-term weather forecasts** and **crop yield predictions**, allowing them to make informed decisions regarding irrigation, planting, and harvesting. For example, if a high-temperature trend is forecasted for the summer months, farmers can adjust irrigation schedules or implement strategies to protect heat-sensitive crops, reducing potential yield losses.

### **Objective Discussion:**

- **Impact and Utility:**
  - This project aims to improve **short-term weather forecasts** and provide actionable insights into the correlation between weather conditions and **crop yield** for **tomatoes** and **almonds** in **Fresno County**. Accurate weather and crop yield predictions will benefit **agriculture** by enabling farmers to optimize irrigation, planting, and harvesting schedules. Additionally, early identification of critical weather conditions (e.g., heatwaves or droughts) will help farmers take timely actions to mitigate crop damage. The project will also provide value by contributing to **agricultural planning** and **disaster preparedness**.

- **Main Claims and Questions:**
  - *Claim 1:* The project will accurately predict short-term weather patterns (e.g., temperature and precipitation) using historical and real-time data for **Fresno County**, enhancing decision-making for farmers and agricultural stakeholders.
  - *Claim 2:* Machine learning models will provide accurate forecasts for how short-term weather patterns (e.g., heatwaves, rainfall) impact **tomato** and **almond** yields, validated through historical crop yield data.
  - *Claim 3:* The project will offer early warnings about potential yield losses based on forecasted critical weather conditions, allowing farmers to take preventive actions to protect crops.

  - *Question 1:* How do short-term weather patterns (e.g., temperature fluctuations, precipitation) influence **tomato** and **almond** yields, and what are the most critical weather variables affecting these yields?
  - *Question 2:* How effective are machine learning models (e.g., LSTM) in predicting short-term weather conditions and their impact on **tomato** and **almond** yields compared to traditional forecasting methods?

## Data Summary and Statistical Analysis

### **Weather-Only Focus:**

#### **1. Time Series Line Plot (Weather Variables Over Time)**
   - **Purpose:** Visualize trends and seasonal patterns in key weather variables over time.
   - **Variables:** Temperature, precipitation, humidity.
   - **Why it’s useful:** It will help you observe how weather conditions change over time and identify patterns, such as seasonal fluctuations or anomalies, across different regions.
   - **Implementation:** Create line plots for different regions in California to observe temperature, precipitation, and humidity trends.
   - **Tools:** Matplotlib, Plotly (for interactive exploration).

#### **2. Heatmap (Seasonal Variations by Region)**
   - **Purpose:** Show how weather variables (e.g., temperature, precipitation) vary across different regions and seasons.
   - **Variables:** Temperature, precipitation.
   - **Why it’s useful:** This will clearly show the regional and seasonal differences in weather patterns, making it easy to identify periods of extreme temperatures or rainfall.
   - **Implementation:** Create a heatmap showing how temperature or precipitation varies monthly across regions (e.g., Coastal, Inland, Central Valley).
   - **Tools:** Seaborn, Plotly.

#### **3. Geographic Map (Spatial Visualization of Weather Data)**
   - **Purpose:** Visualize the spatial distribution of weather data (e.g., average temperature or precipitation) across California.
   - **Variables:** Temperature, precipitation.
   - **Why it’s useful:** A geographic map will help you visualize how weather patterns vary spatially and highlight regional differences.
   - **Implementation:** Use a choropleth map to show the spatial distribution of temperature or precipitation across different agricultural regions.
   - **Tools:** Plotly, Folium, Geopandas.

### **Weather and Crop Yield Data:**

#### **4. Scatter Plot (Weather Variables vs. Crop Yield by Region)**
   - **Purpose:** Explore the relationship between key weather variables and crop yield.
   - **Variables:** Temperature, precipitation, crop yield.
   - **Why it’s useful:** This plot will help you identify how weather conditions (e.g., extreme heat or drought) impact crop yields in different regions.
   - **Implementation:** Plot temperature or precipitation against crop yield for different regions, using separate markers for each crop.
   - **Tools:** Matplotlib, Seaborn.

#### **5. Correlation Matrix Heatmap (Weather Variables and Crop Yield)**
   - **Purpose:** Show correlations between weather variables (e.g., temperature, precipitation) and crop yield.
   - **Variables:** Temperature, precipitation, humidity, crop yield.
   - **Why it’s useful:** This will help identify which weather factors have the strongest impact on specific crop yields, offering valuable insights into climate-crop relationships.
   - **Implementation:** Generate a heatmap to visualize correlations between weather variables and crop yields for selected crops (e.g., almonds, tomatoes, grapes, lettuce).
   - **Tools:** Seaborn, Plotly.

#### **6. Anomaly Detection Plot (Weather Extremes and Yield Impacts)**
   - **Purpose:** Identify and visualize extreme weather events (e.g., heatwaves, droughts) and their impact on crop yields.
   - **Variables:** Temperature, precipitation, crop yield.
   - **Why it’s useful:** This plot will help you highlight how extreme weather conditions directly impact crop productivity and identify patterns of risk.
   - **Implementation:** Use a time series plot to overlay weather anomalies (e.g., Z-scores for temperature) with crop yield data to show how extreme weather events affect yields.
   - **Tools:** Matplotlib, Plotly.

### Preprocessing Steps

**OpenWeatherMap API:**
- Retrieve real-time weather data for specified California regions, capturing key variables such as temperature, precipitation, humidity, and atmospheric pressure.
- Fetch historical weather data from 1979 to the present, allowing comparison with real-time data.
- No preprocessing will be done on dates; date parsing will remain flexible for time interval analysis.
- The data will be aggregated by date, city, and other relevant attributes such as temperature, humidity, and precipitation.

**NOAA Climate Data:**
- Download historical weather records focusing on temperature and precipitation from 1979 to the present for California’s agricultural regions.
- Include relevant seasonal indicators such as temperature patterns and rainfall amounts during critical crop growth periods.
- No specific date preprocessing; historical weather records will align with crop yield data for later correlation analysis.

**USDA Crop Yield Data:**
- Collect historical crop yield data for key crops (almonds, tomatoes, grapes, lettuce) from 1979 to the present.
- Align the crop yield data with the corresponding weather conditions in the same regions and timeframes for correlation analysis.
- Regions are assigned based on counties to facilitate regional crop yield comparisons with weather data.

**Region Assignment for Crop Yield Data:**
- Assign each county in the dataset to a predefined region (e.g., San Francisco, Riverside, Fresno) based on its geographical location.
- Use these regions to aggregate and filter crop data by year and region, grouping by crop name to derive relevant metrics such as harvested acres, yield, production, price per unit, and overall value.

**Weather Data Aggregation:**
1. **Daily Aggregation:**
   - Parse and clean the datetime column, removing unnecessary time zone information.
   - Aggregate weather data (temperature, pressure, humidity, etc.) daily, grouped by both date and city, to get a daily summary of weather conditions per city.

2. **Monthly Aggregation:**
   - Extract the year and month from the datetime column.
   - Group the weather data by city and month to generate monthly summaries for each location, including average temperature, humidity, and precipitation.

3. **Yearly Aggregation:**
   - Extract the year from the datetime column.
   - Group the weather data by city and year to produce yearly summaries, focusing on the same variables as daily and monthly aggregations.

Each step produces a CSV output that will serve as an input for the next phase of analysis. This approach maintains flexibility by not heavily preprocessing dates, enabling more dynamic analysis during the model-building phase. The preprocessing pipeline ensures consistent, clean data for later correlation and time-series analysis.

<!--- 
----------
The following sections should be used for the full proposal document. These are not required for the proposal draft discussion.
----------
-->


## **Basic Data Properties and Analysis Techniques**

### **1. Data Overview and Summary Statistics**

**Objective:**
To provide an initial understanding of both weather and crop yield datasets by summarizing key characteristics and overall distributions of the data.

**Techniques:**
- **Descriptive Statistics:**
  - **Mean, Median, and Mode:** Calculate these central tendency measures for continuous variables such as temperature, precipitation, humidity, and crop yield to understand the average conditions and productivity.
  - **Standard Deviation and Variance:** Measure the variability of weather variables and crop yield to assess the spread and dispersion of the data.
  - **Min and Max Values:** Identify the range of weather conditions and crop yield values to understand the extremes observed in the dataset.

- **Distribution Analysis:**
  - **Histograms:** Create histograms to visualize the frequency distribution of weather variables and crop yield (e.g., temperature distribution across regions or yield distribution for each crop).
  - **Box Plots:** Use box plots to visualize the spread, central tendency, and potential outliers in both weather data and crop yield data.

### **2. Temporal Analysis**

**Objective:**
To analyze how weather variables change over time and assess how these changes correlate with crop yield during different growing seasons.

**Techniques:**

- **Seasonal Decomposition (STL Decomposition):**
  - **Implementation Plan:** Use STL decomposition to break down weather data (e.g., temperature, precipitation) into seasonal, trend, and residual components.
    - Select appropriate timeframes for weather data (e.g., daily or monthly) and crop yield data (e.g., seasonal) to capture seasonal variations.
    - Apply **STL decomposition** to visualize and analyze how seasonal weather patterns correspond with crop productivity (e.g., rainfall during growing seasons correlating with higher yields).
    - Assess regional variations (e.g., Coastal, Inland, Central Valley) to determine how different regions experience unique seasonal patterns that impact crop yield.

- **Autoregressive Integrated Moving Average (ARIMA):**
  - **Implementation Plan:** Fit ARIMA models to forecast short-term weather conditions.
    - Conduct **exploratory data analysis (EDA)** to determine the appropriate differencing for stationarity of the time series data.
    - Use **ACF** and **PACF** plots to identify the optimal order for the ARIMA model (p, d, q).
    - Train the ARIMA model on historical weather data, focusing on short-term predictions of temperature and precipitation.
    - Validate the model’s performance using metrics like **Mean Absolute Error (MAE)** and compare predicted weather values with real-time data.
    - Integrate crop yield data into the model to predict how upcoming weather conditions might influence yields.

### **3. Anomaly Detection**

**Objective:**
To identify unusual or extreme weather events (e.g., heatwaves, droughts) that deviate significantly from historical norms and assess their impact on crop yield.

**Techniques:**

- **Z-Score Analysis:**
  - **Implementation Plan:** 
    - Calculate the mean and standard deviation of weather variables (e.g., temperature, precipitation) from historical data.
    - Compute **Z-scores** for each weather observation to measure the number of standard deviations away from the mean.
    - Identify anomalies (e.g., extreme temperatures or rainfall) by setting a threshold (e.g., Z-scores exceeding ±3).
    - Overlay weather anomalies with crop yield data to assess whether extreme weather events had significant effects on yield.

- **Interquartile Range (IQR) Method:**
  - **Implementation Plan:**
    - Calculate the **first (Q1)** and **third quartiles (Q3)** of weather data to compute the **IQR**.
    - Define upper and lower bounds (e.g., Q1 - 1.5 * IQR and Q3 + 1.5 * IQR) to detect weather outliers (e.g., extreme rainfall or drought).
    - Visualize outliers using box plots and examine how these anomalies correlate with crop yield drops during specific growing seasons.

### **4. Correlation Analysis Between Weather and Crop Yield**

**Objective:**
To explore the relationships between weather variables (e.g., temperature, precipitation) and crop yield to identify how short-term weather fluctuations affect agricultural productivity.

**Techniques:**
- **Pearson and Spearman Correlation Analysis:**
  - Use **Pearson correlation** to assess linear relationships and **Spearman correlation** for non-linear relationships between weather variables and crop yield.
  - Identify which weather factors (e.g., temperature, rainfall) most strongly influence yield for each crop (e.g., almonds, tomatoes, grapes).

- **Scatter Plots:**
  - Visualize relationships between weather variables (e.g., temperature vs. precipitation) and crop yield to detect patterns or thresholds where yield is significantly impacted.

### **5. Data Completeness and Integrity**

**Objective:**
To ensure the quality and consistency of weather and crop yield data for accurate analysis and predictions.

**Techniques:**
- **Missing Value Analysis:**
  - Analyze patterns of missing values in both weather and crop yield data to understand their distribution and potential impact on analysis.
  - Use appropriate **imputation strategies** (e.g., forward/backward fill for time series data) to handle missing data and minimize its effect on results.

- **Data Consistency Checks:**
  - Perform **data validation** by cross-referencing weather data with known historical trends and comparing crop yield data against established benchmarks for accuracy.
  - Ensure consistency across datasets (e.g., weather data from different sources or crop yield data for different regions) to avoid discrepancies in analysis.


## Automation, Scalability, and Portability

### **1. Automation**

**Objective:**
To streamline data processing and minimize manual intervention, allowing for efficient updates and analysis.

**Techniques:**
- **Scheduled Data Retrieval:** Utilize Heroku Scheduler or cron jobs to automate the regular fetching of new weather data from APIs, ensuring the system is continuously updated with the latest information.
- **Automated Data Processing Pipelines:** Implement ETL processes using MongoDB to efficiently manage and transform incoming data, ensuring consistency and reducing processing time.
- **Error Handling:** Establish comprehensive logging and alert systems to monitor data processing activities, enabling rapid identification and resolution of any issues that arise.

### **2. Scalability**

**Objective:**
To design the system to handle increasing data volumes and user demands without compromising performance.

**Techniques:**
- **Cloud Services:** Leverage Heroku’s scalable infrastructure to dynamically adjust resources based on user demand and data volume, ensuring optimal performance during peak usage.
- **Database Optimization:** Utilize MongoDB's flexible schema and indexing capabilities to manage large datasets efficiently, allowing for quick queries and data retrieval.
- **Load Balancing:** Implement load balancing techniques to distribute incoming traffic evenly across multiple instances, ensuring consistent performance and responsiveness under heavy loads.

### **3. Portability**

**Objective:**
To ensure the system can be easily adapted to various environments with minimal modifications.

**Techniques:**
- **Containerization:** Use Docker to containerize the application and its dependencies, ensuring consistent behavior across development, staging, and production environments.
- **Configuration Management:** Utilize environment variables and configuration files to manage environment-specific settings, facilitating easy deployment in different contexts without hardcoding values.
- **Documentation:** Maintain detailed documentation of the system architecture, setup procedures, and dependencies, alongside version control (e.g., Git), to streamline knowledge transfer and adaptation for future developers.

### **4. Future Considerations**

**Objective:**
To prepare for ongoing enhancements and the integration of new features as project requirements evolve.

**Techniques:**
- **Modular Design:** Structure the system with modular components that can be independently updated or replaced, allowing for easy integration of new features without disrupting existing functionality.
- **Extensible Architecture:** Develop well-defined APIs and interfaces that facilitate the addition of new data sources or analytical methods, ensuring the system can adapt to emerging needs and technologies.

<!--- 
----------
The following sections should be used for the analysis planning. These are not required for the proposal document submission.
----------
-->

## Data Analysis and Algorithms

### Weather Data Model Usage

#### Types of (Advanced) Analysis Conducted:
This analysis leverages a combination of traditional machine learning techniques and advanced time series models to predict **temperature** and **crop-related outcomes** using **weather data**.

1. **Data Preprocessing and Feature Selection**:
   - **Weather data** was preprocessed by selecting relevant features such as **temperature**, **dew point**, **humidity**, and **wind speed**, among others.
   - Missing values were handled by filling them with the column-wise mean, ensuring the dataset is ready for model training.
   
2. **Model Training and Evaluation**:
   - Several models, including **Random Forest**, **Gradient Boosting**, **Linear Regression**, and **Support Vector Regressor (SVR)**, were trained and evaluated.
   - The models were assessed based on **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R²** metrics to understand their performance.
   - Model training was automated, and the best-performing model was saved for future use.

3. **Visualization and Residual Analysis**:
   - **Actual vs. Predicted** scatter plots were generated for each model, allowing for a visual comparison of model predictions.
   - **Residual plots** were created to inspect the errors made by the models.
   - Performance metrics were summarized in a bar plot to provide an easy comparison between models.

4. **Time Series Forecasting with Prophet**:
   - **Prophet** was used to forecast temperature based on historical weather data. This approach was applied on a **city-by-city** basis.
   - Performance metrics were computed for each city, allowing us to analyze how accurately the model could predict temperatures for specific locations.
   - Future predictions were made using Prophet, with visualizations that showed the forecasted temperature trends.

5. **LSTM for Sequential Data Prediction**:
   - **Long Short-Term Memory (LSTM)** networks were used to capture sequential dependencies in weather data, particularly to predict temperature.
   - The LSTM model was trained using sequences of weather data (with a lookback of 60 time steps), and performance metrics such as **MAE**, **RMSE**, and **R²** were calculated to evaluate the predictions.
   - **Dropout layers** were incorporated in the LSTM model to prevent overfitting.

---

### Explanation of Algorithmic Properties and Logic

#### 1. **Random Forest and Gradient Boosting**:
   - **Inputs**: Selected weather variables (`temp`, `humidity`, `wind_speed`, etc.).
   - **Outputs**: Predictions of temperature based on historical weather features.
   - **Logic**: 
     - **Random Forest** aggregates predictions from multiple decision trees, reducing variance and improving prediction accuracy.
     - **Gradient Boosting** iteratively trains models by focusing on correcting errors made by previous models, resulting in improved performance.
   - **Major Findings**: These models provided robust predictions with minimal overfitting, as seen from their test set performance.

#### 2. **Support Vector Regressor (SVR)**:
   - **Inputs**: Same weather variables.
   - **Outputs**: Predictions of temperature.
   - **Logic**: **SVR** attempts to find the hyperplane that best fits the data points in the feature space, minimizing the error margin for regression.
   - **Major Findings**: SVR, while accurate, tended to underperform compared to Random Forest and Gradient Boosting due to its sensitivity to outliers.

#### 3. **Prophet Time Series Forecasting**:
   - **Inputs**: Date (`ds`), temperature values (`y`), and weather features.
   - **Outputs**: Forecasted temperature values for future dates.
   - **Logic**: **Prophet** decomposes time-series data into trend and seasonality components, using historical weather patterns to predict future temperatures.
   - **Major Findings**: The model accurately captured seasonal trends in temperature data and provided reliable forecasts for cities like Fresno.

#### 4. **Long Short-Term Memory (LSTM)**:
   - **Inputs**: Sequences of weather data (e.g., `temp`, `humidity`).
   - **Outputs**: Temperature predictions.
   - **Logic**: **LSTM** networks are designed to capture long-term dependencies in sequential data, making them particularly useful for time-series forecasting. LSTMs use memory cells to store relevant information over time, allowing for the prediction of future values.
   - **Major Findings**: LSTM models showed strong predictive performance, especially for capturing subtle temporal dependencies in weather data.

---

### Inputs and Outputs for Each Algorithm:

1. **Random Forest, Gradient Boosting, Linear Regression, SVR**:
   - **Inputs**: Weather features (e.g., temperature, humidity, wind speed, etc.).
   - **Outputs**: Predictions for the target variable (temperature).
   - **Major Findings**: Gradient Boosting provided the most accurate predictions, while Random Forest also performed well with reduced overfitting.

2. **Prophet**:
   - **Inputs**: Date (`ds`), temperature values (`y`), and optional weather regressors.
   - **Outputs**: Forecasted temperature values.
   - **Major Findings**: Prophet captured trends and seasonality effectively, making accurate predictions for future temperatures.

3. **LSTM**:
   - **Inputs**: Sequences of weather data over time.
   - **Outputs**: Predictions for temperature.
   - **Major Findings**: LSTM showed good performance in predicting future weather variables by learning from long-term dependencies in sequential weather data.

---

### Expected Major Findings:

- **Traditional Models**: Gradient Boosting and Random Forest are expected to provide accurate temperature predictions based on weather features.
- **Prophet and LSTM**: Both models will provide robust time-series predictions for temperature, capturing seasonal and trend-based changes in weather patterns.

### Crop Data Model Usage

#### Types of Analysis Conducted:
1. **Data Preprocessing and Resampling**:
   - Crop data was **resampled** from yearly to daily intervals using **linear interpolation** to fill in missing data points.
   - Weather data for **Fresno** was filtered to match the date range of interest (1980-2020) for further analysis and merged with crop data.

2. **Exploratory Data Analysis (EDA)**:
   - Time-series plots were generated for each feature in the crop dataset to visually explore relationships over time.

3. **Feature Selection and Model Training**:
   - A **feature selection methodology** was implemented using **Recursive Feature Elimination (RFE)** to automatically select the most relevant weather features for crop prediction.
   - Several **candidate models** (Gradient Boosting, Linear Regression) were trained using the selected features.

4. **Time Series Forecasting**:
   - Two advanced time series models, **Prophet** and **LSTM**, were applied to the dataset to capture the temporal dynamics between weather and crop variables over time.

### Explanation of Algorithmic Properties and Logic

#### 1. **Data Preprocessing and Resampling**
   - **Inputs**: Crop data (Yearly), Weather data (Daily).
   - **Outputs**: Resampled daily crop data, merged with weather data by matching dates.
   - **Description**: The crop data was resampled from yearly to daily frequency using linear interpolation, filling missing data points. The weather data was filtered to include only dates and regions relevant to crop analysis.

#### 2. **Time Series Plots for EDA**
   - **Inputs**: Resampled daily crop data, weather data.
   - **Outputs**: Time series plots for each variable.
   - **Description**: Visual plots were generated for each crop variable to inspect the trends over time and the relationship between weather conditions and crop outcomes.

#### 3. **Feature Selection and Model Training**
   - **Inputs**: Weather features (`temp`, `humidity`, `wind_speed`, etc.), crop targets (`Yield`, `Production`, `Harvested Acres`).
   - **Outputs**: Trained models (Random Forest, Gradient Boosting, Linear Regression) with optimal hyperparameters.
   - **Description**:
     - Feature selection was performed using **RFE** with **Random Forest** as the base estimator. This method selects the most relevant features for each model.
     - **GridSearchCV** was used to optimize hyperparameters and select the best-performing model using **cross-validation**.
     - Each model was evaluated on the test set using performance metrics such as **RMSE** and **R²**.

#### 4. **Prophet Time Series Forecasting**
   - **Inputs**: Date (`ds`), crop variables (`y`), weather features as additional regressors.
   - **Outputs**: Time-series forecasts for crop variables, performance metrics (RMSE, R²).
   - **Description**: 
     - The **Prophet** model was trained for each crop target (Yield, Production, Harvested Acres) using weather features as external regressors. 
     - Prophet handles seasonality, trends, and missing data effectively, making it suitable for predicting agricultural variables that vary cyclically over time.
     - The model was evaluated using standard metrics such as **RMSE** to assess its forecasting ability.

#### 5. **LSTM for Time Series Prediction**
   - **Inputs**: Sequences of weather data over time (`X`), crop targets (`y`).
   - **Outputs**: LSTM model predictions, performance metrics (RMSE, R²).
   - **Description**:
     - The **LSTM** model was trained using sequences of weather variables, with each sequence predicting future crop yields. LSTMs excel at capturing long-term dependencies in time-series data.
     - The LSTM architecture included two layers of LSTM units, followed by fully connected layers to output the crop predictions.

### Advanced Algorithms and Logic

#### 1. **Recursive Feature Elimination (RFE)**:
   - **Logic**: RFE recursively removes the least important features based on the model’s performance, refining the feature set for the next iteration.
   - **Inputs**: All weather variables (`temp`, `humidity`, `wind_speed`, etc.).
   - **Outputs**: The best set of features for predicting crop variables.
   - **Benefits**: RFE ensures that only the most relevant features are used for model training, reducing model complexity and improving performance.

#### 2. **GridSearchCV**:
   - **Logic**: It performs an exhaustive search over specified hyperparameter values for each model, using cross-validation to find the best combination of hyperparameters.
   - **Inputs**: Hyperparameter grid, model pipeline.
   - **Outputs**: The best model with optimized hyperparameters.
   - **Benefits**: Ensures that the models are tuned for optimal performance by evaluating different hyperparameter configurations.

#### 3. **Prophet**:
   - **Logic**: Prophet decomposes time-series data into trend, seasonality, and holiday effects while also allowing external regressors (in this case, weather features).
   - **Inputs**: Time (`ds`), crop target (`y`), weather regressors.
   - **Outputs**: Time-series forecasts for each crop variable.
   - **Benefits**: Prophet is robust to missing data and can handle seasonal and trend-based changes in time series, making it highly suitable for agricultural forecasting.

#### 4. **LSTM (Long Short-Term Memory)**:
   - **Logic**: LSTM networks use memory cells to store information over long periods, capturing temporal dependencies in time-series data.
   - **Inputs**: Sequences of weather features.
   - **Outputs**: Predictions for crop variables.
   - **Benefits**: LSTM is particularly powerful for sequential data like time-series weather and crop data, as it can capture both short- and long-term dependencies.

### Inputs and Outputs for Each Algorithm

1. **Traditional Models (Random Forest, Gradient Boosting, Linear Regression)**:
   - **Inputs**: Weather features (e.g., temperature, humidity), crop target variables.
   - **Outputs**: Predictions for crop targets (`Yield`, `Production`, `Harvested Acres`).

2. **Prophet**:
   - **Inputs**: Date (`ds`), crop targets, weather features as regressors.
   - **Outputs**: Forecasts for crop targets.

3. **LSTM**:
   - **Inputs**: Sequences of weather features, crop target variables.
   - **Outputs**: Predictions for future crop variables.

### Expected Major Findings:
- **Traditional Models**: Identify the most important weather factors influencing crop yields.
- **Prophet and LSTM**: Forecast crop production and yield based on weather patterns and trends, providing insights into seasonal variations and long-term weather impacts on agriculture.


<!--- 
----------
The following sections should be used for the analysis outcome presentation. These are not required for the analysis plan submission.
----------
-->
# Analysis Outcomes
<!--- Explain the analysis you conducted and show the results. Discuss how the data, your analysis, and/or visualization can support the claims or findings. What will be the recommendations or suggestions you can make based on the results? Use bullet points, tables, and figures (if possible) to increase the readability of the document. -->



<!--- 
----------
The following sections should be used for the visualization planning. These are not required for the analysis outcome presentation.
----------
-->


# Visualization
## Visualization Plan
<!--- List and explain what types of plots you plan to provide and what you are trying to show through the diagrams. You should explore the best way to visualize the information and message based on the lectures on visualization and related topics. It is required to have at least two interactive graphing and five static plots. -->

## Web Page Plan
<!--- Explain how many pages you will have in total and what content will be shown in each page. (Each diagram discussed above should be given a proper location in this section. Also, it is required to have (1) "Project Objective" page, which explains the main goals and data sources, and (2) "Analytical Methods" page, where you explain the major techniques used in the project and provide further references. -->
