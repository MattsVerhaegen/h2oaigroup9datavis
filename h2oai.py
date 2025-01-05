import sqlite3
import pandas as pd
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator
import matplotlib.pyplot as plt

# Start H2O instance
h2o.init()

# Connect to database file
data = sqlite3.connect("database_datavis.db")

# Query the table
emissions_query = "SELECT * FROM emissions"

# Load data into DataFrames
emissions = pd.read_sql_query(emissions_query, data)

# Close the connection
data.close()

# Create a year column 
emissions = emissions[pd.to_datetime(emissions['Date'], format='%d-%m-%Y', errors='coerce').notnull()]
emissions['Year'] = pd.to_datetime(emissions['Date']).dt.year

# Calculate mean CO2 emissions per year
mean_emissions = emissions.groupby('Year')['Metric_Tons_Per_Capita'].mean().reset_index()

# Convert the dataframe to H2O frame
h2o_df = h2o.H2OFrame(emissions)

# Specify the response and predictor variables
y = 'Metric_Tons_Per_Capita'
x = ['Year']

# Train the model using H2O Linear Regression
prediction_model = H2OGeneralizedLinearEstimator()
prediction_model.train(x=x, y=y, training_frame=h2o_df)

# View the model performance
performance = prediction_model.model_performance()
print(performance)

# Predict future CO2 emissions for year 2020 to 2040
future_years = pd.DataFrame({'Year': range(2020, 2041)})
future_predictions = []

# Predict for every year in future years
for year in future_years['Year']:
    future_year_df = pd.DataFrame({'Year': [year]})
    future_h2o_df = h2o.H2OFrame(future_year_df)
    prediction = prediction_model.predict(future_h2o_df)
    future_predictions.append(prediction.as_data_frame().iloc[0, 0])

future_years['Predicted_CO2_Emissions'] = future_predictions

# Plot the historical and predicted CO2 emissions
plt.figure(figsize=(10, 6))
plt.plot(mean_emissions['Year'], mean_emissions['Metric_Tons_Per_Capita'], label='Historical CO2 Emissions')
plt.plot(future_years['Year'], future_years['Predicted_CO2_Emissions'], label='Predicted CO2 Emissions', linestyle='--')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.title('Historical and Predicted CO2 Emissions')
plt.legend()
plt.show()

# End H2O instance
h2o.shutdown()