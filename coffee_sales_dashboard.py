import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html
import plotly.graph_objects as go

#Load data
df = pd.read_csv("coffee_sales.csv")

#Convert date to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

#Extract hour and week
df["day_of_week"] = df["date"].dt.dayofweek
df["hour"] = 12  #Assume purchases are made at noon

#Change each coffee type to a number label
label_encoder = LabelEncoder()
df["coffee_id"] = label_encoder.fit_transform(df["coffee_name"]) + 1  #1 to N

#Ensure money column is a float
df["money"] = df["money"].astype(float)

#Extract the year
df["year"] = df["date"].dt.year

predictions = {}

for coffee_name in df["coffee_name"].unique():
    #Filter data for this specific coffee type
    df_coffee = df[df["coffee_name"] == coffee_name]

    #Define independent and dependent variables for this coffee type
    x = df_coffee[["hour", "day_of_week", "coffee_id", "year"]]
    y = df_coffee["money"]

    #Create and train the model
    model = LinearRegression()
    model.fit(x, y)

    #Forecast for each year
    years = list(range(2024, 2029)) 
    coffee_predictions = []
    for year in years:
        x_new = pd.DataFrame({
            "hour": [12],  
            "day_of_week": [2], 
            "coffee_id": [df_coffee["coffee_id"].iloc[0]],
            "year": [year]
        })
        y_pred_year = model.predict(x_new)
        coffee_predictions.append(y_pred_year[0])

    predictions[coffee_name] = {
        "years": years,
        "predictions": coffee_predictions
    }

#Create a Dash app
app = dash.Dash(__name__)

#Create figures for the dashboard
fig = go.Figure()

#Add traces for each coffee type
for coffee_name, data in predictions.items():
    fig.add_trace(go.Scatter(x=data["years"], y=data["predictions"], mode='lines', name=coffee_name))

#Update the layout for better presentation
fig.update_layout(
    title="Sales Forecast by Coffee Type",
    xaxis_title="Year",
    yaxis_title="Predicted Sales ($)",
    legend_title="Coffee Types",
    template="plotly_dark"
)

#Define the layout of the app
app.layout = html.Div(children=[
    html.H1("Coffee Sales Forecast Dashboard"),
    dcc.Graph(figure=fig),
])

#Run the Dash app 
if __name__ == '__main__':
    app.run(debug=True) 

