import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import matplotlib.pyplot as plt

# Start H2O instance
h2o.init()

# Load datasets
life_expectancy = pd.read_csv("life_expectancy.csv")
emissions = pd.read_csv("emissions.csv")

# Merge datasets on common keys (e.g., Country, Year)
life_expectancy['Year'] = life_expectancy['Year'].astype(int)
emissions['Year'] = pd.to_datetime(emissions['Date']).dt.year
merged_data = pd.merge(life_expectancy, emissions, 
                        left_on=['Entity', 'Year'], 
                        right_on=['Country', 'Year'])

# Select relevant columns
merged_data = merged_data[['Entity', 'Region', 'Year', 'Life_expectancy', 'Metric_Tons_Per_Capita']]
merged_data = merged_data.dropna()

# Upload data to H2O
h2o_data = h2o.H2OFrame(merged_data)

# Define predictors and response variable
x = ['Year', 'Metric_Tons_Per_Capita']
y = 'Life_expectancy'

# Train AutoML model
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=h2o_data)

# View leaderboard
lb = aml.leaderboard
print(lb)

# Get the best model
best_model = aml.leader

# Predict on new data (optional)
predictions = best_model.predict(h2o_data)
h2o_data['Predicted_Life_Expectancy'] = predictions

# Save the model and export predictions
model_path = h2o.save_model(model=best_model, path="./best_model", force=True)
h2o_data.as_data_frame().to_csv("predicted_life_expectancy.csv", index=False)

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['Metric_Tons_Per_Capita'], merged_data['Life_expectancy'], alpha=0.5)
plt.title("Relationship between CO2 Emissions and Life Expectancy")
plt.xlabel("Metric Tons Per Capita")
plt.ylabel("Life Expectancy")
plt.show()

# Shutdown H2O instance
h2o.shutdown(prompt=False)