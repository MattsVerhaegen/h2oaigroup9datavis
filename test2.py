import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Start H2O instance
h2o.init()

# Connect to database file
data = sqlite3.connect("database_datavis.db")

# Query the tables
life_expectancy_query = "SELECT * FROM life_expectancy"
emissions_query = "SELECT * FROM emissions"

# Load data into DataFrames
life_expectancy = pd.read_sql_query(life_expectancy_query, data)
emissions = pd.read_sql_query(emissions_query, data)

# Close the connection
data.close()

# Ensure 'Year' column in life_expectancy contains only numeric values
life_expectancy = life_expectancy[pd.to_numeric(life_expectancy['Year'], errors='coerce').notnull()]
life_expectancy['Year'] = life_expectancy['Year'].astype(int)

# Ensure 'Date' column in emissions contains only valid date strings
emissions = emissions[pd.to_datetime(emissions['Date'], errors='coerce').notnull()]
emissions['Year'] = pd.to_datetime(emissions['Date']).dt.year

# Merge datasets on related columns and change date to only year
merged_data = pd.merge(life_expectancy, emissions, 
                        left_on=['Entity', 'Year'], 
                        right_on=['Country', 'Year'])

# Select relevant columns
merged_data = merged_data[['Entity', 'Region', 'Year', 'Life_expectancy', 'Metric_Tons_Per_Capita']]
# Remove missing values
merged_data = merged_data.dropna()

# Filter on Belgium
country_data = merged_data[merged_data['Entity'] == 'Belgium']

# Upload data to H2O
h2o_data = h2o.H2OFrame(merged_data)

# Define predictors and response variable
x = ['Year', 'Metric_Tons_Per_Capita']
y = 'Life_expectancy'

# Train Linear Regression model
linear_model = H2OGeneralizedLinearEstimator(family="gaussian")
linear_model.train(x=x, y=y, training_frame=h2o_data)

# Check the performance of the model
print(linear_model.model_performance(h2o_data))

# Projected emission of 2040 in Belgium
#https://www.rtl.nl/nieuws/artikel/5433474/nieuw-klimaatdoel-2040#:~:text=De%20Europese%20Commissie%20heeft%20vandaag,worden%2C%20zo%20is%20het%20idee.
# -90% of 1990 emission
emission_2040 = 0.1 * merged_data.loc[merged_data['Year'] == 1990, 'Metric_Tons_Per_Capita'].values[0]

# Predict for future years 2025 and 2040
future_data = pd.DataFrame({
    'Year': [2019, 2040],
    'Metric_Tons_Per_Capita': [8.1, emission_2040]  # Use the predicted value for 2040
})
future_h2o = h2o.H2OFrame(future_data)
future_predictions = linear_model.predict(future_h2o)
# Add predictions to future data
future_data['Predicted_Life_Expectancy'] = future_predictions.as_data_frame().values.flatten()
print(future_data)

# Save the model and export predictions
model_path = h2o.save_model(model=linear_model, path="./linear_model", force=True)
future_data.to_csv("future_life_expectancy_predictions.csv", index=False)

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(country_data['Year'], country_data['Life_expectancy'], label="Observed", alpha=0.5)
plt.plot(future_data['Year'], future_data['Predicted_Life_Expectancy'], label="Predicted", color='red')
plt.title("Life Expectancy Predictions for Belgium")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.legend()
plt.show()

# Shutdown H2O instance
h2o.shutdown(prompt=False)