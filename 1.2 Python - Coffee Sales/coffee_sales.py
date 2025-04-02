
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Load and preprocess the dataset
df = pd.read_csv("coffee_sales.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year

#Map coffee types to numeric labels
label_encoder = LabelEncoder()
df["coffee_id"] = label_encoder.fit_transform(df["coffee_name"]) + 1

#Calculate number of sales for each class
# Group by coffee name  and year to get the number of sales
df_sales = df.groupby(["coffee_name", "year"]).size().reset_index(name="num_sales")

model = LinearRegression()

#Set 5 year range for prediction
years = np.arange(2024, 2029)
predictions = {}

#Filter data for the coffee type
for coffee_name in df_sales["coffee_name"].unique():
    df_coffee = df_sales[df_sales["coffee_name"] == coffee_name]
    X = df_coffee[["year"]]  
    y = df_coffee["num_sales"] #number of sales is the target

    #Train the model and predict
    model.fit(X, y)
    coffee_predictions = [model.predict([[year]])[0] for year in years]
    
    predictions[coffee_name] = coffee_predictions

#Plot the predictions
plt.figure(figsize=(6, 6))
for coffee_name, pred in predictions.items():
    plt.plot(years, pred, label=coffee_name)

plt.title("Sales Forecast by Coffee Type (Number of Transactions)")
plt.xlabel("Year")
plt.ylabel("Predicted Number of Sales")
plt.legend(title="Coffee Types")
plt.grid(True)
plt.xticks(years)
plt.show()



