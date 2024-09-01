# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 00:13:06 2024

@author: duongj14
"""

import gzip
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def read_flight_data(filepath):
    with gzip.open(filepath, 'rt') as f:
        df = pd.read_csv(f)
    return df
    
def get_weather(date, lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://customer-archive-api.open-meteo.com/v1/archive"
    params = {
    	"latitude": lat,
    	"longitude": lon,
    	"start_date": str(date),
    	"end_date": str(date),
    	"hourly": ["precipitation", "cloud_cover", "wind_speed_100m", "wind_direction_100m"],
        "apikey": "3D6uPmqBfuA7iixG"
    }
    response = openmeteo.weather_api(url, params=params)
    return response[0]

def ensure_landing(group):
    if group.iloc[-1]['Flight Level'] != 0:
        group.iloc[-1, group.columns.get_loc('Flight Level')] = 0
    return group

def train_and_evaluate_model(flight_df, weather_df):
    # Convert datetime columns to datetime dtype
    flight_df['FILED ARRIVAL TIME'] = pd.to_datetime(flight_df['FILED ARRIVAL TIME'])
    flight_df['ACTUAL ARRIVAL TIME'] = pd.to_datetime(flight_df['ACTUAL ARRIVAL TIME'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    weather_df['Hour'] = pd.to_datetime(weather_df['Hour'])

    merged_df = pd.merge(flight_df, weather_df, left_on=['ECTRL ID', 'ADEP Latitude', 'ADEP Longitude'], right_on=['ECTRL ID', 'Latitude', 'Longitude'], how='left')
    merged_df = merged_df.drop(columns=['Latitude', 'Longitude'])
    merged_df['Time Difference'] = (merged_df['ACTUAL ARRIVAL TIME'] - merged_df['FILED ARRIVAL TIME']).dt.total_seconds()

    feature_columns = ['ADEP Latitude', 'ADEP Longitude', 'ADES Latitude', 'ADES Longitude', 'Requested FL', 'Actual Distance Flown (nm)', 'precipitation', 'cloud_cover', 'wind_speed_100m', 'wind_direction_100m']
    X = merged_df[feature_columns]
    y = merged_df['Time Difference']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    mse = root_mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return regressor, feature_columns

def predict_actual_arrival_time(regressor, feature_columns, filed_arrival_time, features):
    features_df = pd.DataFrame([features], columns=feature_columns)
    time_difference = regressor.predict(features_df)[0]
    predicted_actual_arrival_time = pd.to_datetime(filed_arrival_time) + pd.to_timedelta(time_difference, unit='s')
    return predicted_actual_arrival_time

cwd = os.getcwd()
datasetDIR = os.path.join('..', 'dataset', '20171201')
dataset_path = os.path.join(cwd, datasetDIR)

flightPoints = 'Flight_Points_Actual_20171201_20171231.csv.gz'
# flightPoints = "test_flight_point_data.csv"
flightPointsPath = os.path.join(dataset_path, flightPoints)

flightClass = 'Flights_20171201_20171231.csv.gz'
# flightClass = "test_flights_data.csv"
flightClassPath = os.path.join(dataset_path, flightClass)

timeSeriesDF = pd.read_csv(flightPointsPath)
flightClassDF = pd.read_csv(flightClassPath)

date_format = '%d-%m-%Y %H:%M:%S'
flightClassDF['ACTUAL ARRIVAL TIME'] = pd.to_datetime(flightClassDF['ACTUAL ARRIVAL TIME'], format=date_format, errors='coerce')
flightClassDF['ACTUAL OFF BLOCK TIME'] = pd.to_datetime(flightClassDF['ACTUAL OFF BLOCK TIME'], format=date_format, errors='coerce')

df_merged = pd.merge(timeSeriesDF, flightClassDF, on='ECTRL ID')

df_merged['Time Over'] = pd.to_datetime(df_merged['Time Over'], format='%d-%m-%Y %H:%M:%S')
df_merged['FILED ARRIVAL TIME'] = pd.to_datetime(df_merged['FILED ARRIVAL TIME'], format='%d-%m-%Y %H:%M:%S')

df_merged['Time Difference'] = (df_merged['FILED ARRIVAL TIME'] - df_merged['Time Over']).dt.total_seconds() / 60

df_merged = df_merged.dropna(subset=['Time Difference', 'ACTUAL ARRIVAL TIME', 'ACTUAL OFF BLOCK TIME', 'Time Over'])
df_merged['Sequence Number'] = df_merged['Sequence Number'].astype(int)
df_merged['Flight Level'] = df_merged['Flight Level'].astype(int)
df_merged['Latitude'] = df_merged['Latitude'].astype(float)
df_merged['Longitude'] = df_merged['Longitude'].astype(float)
df_merged['ADEP Latitude'] = df_merged['ADEP Latitude'].astype(float)
df_merged['ADEP Longitude'] = df_merged['ADEP Longitude'].astype(float)
df_merged['ADES Latitude'] = df_merged['ADES Latitude'].astype(float)
df_merged['ADES Longitude'] = df_merged['ADES Longitude'].astype(float)
df_merged['Day of the Week'] = df_merged['Time Over'].dt.dayofweek
df_merged['Month'] = df_merged['Time Over'].dt.month
df_merged['Hour'] = df_merged['Time Over'].dt.floor('h').dt.hour
df_merged['Minute'] = df_merged['Time Over'].dt.minute
df_merged['Second'] = df_merged['Time Over'].dt.second

df_cleaned = df_merged.dropna()

# X = df_cleaned[['Sequence Number', 'Flight Level', 'Latitude', 'Longitude', 'ADEP Latitude', 'ADEP Longitude', 'ADES Latitude', 'ADES Longitude','Day of the Week','Month','Hour','Minute','Time Over','ECTRL ID']]
# y = df_cleaned['Time Difference'].to_frame()

X = df_cleaned[['Sequence Number', 'Flight Level', 'Latitude', 'Longitude', 'ADEP Latitude', 'ADEP Longitude', 'ADES Latitude', 'ADES Longitude','Day of the Week','Month','Hour','Minute','Time Over','ECTRL ID']][:750010]
y = df_cleaned['Time Difference'][:750010].to_frame()

# X = z.iloc[:10,:].copy()
# X_copy = X.iloc[:100,:].copy()
# # y_copy = y.iloc[:100].copy()

unique_ectrl_ids = X["ECTRL ID"].unique()
flightClassfiltered_df = flightClassDF[flightClassDF["ECTRL ID"].isin(unique_ectrl_ids)]

unique_combinations = X[['ECTRL ID','Time Over', 'Latitude', 'Longitude']].drop_duplicates()
unique_combinations['Time Over'] = unique_combinations['Time Over'].dt.tz_localize('UTC')

weather_data_list = []
matched_rows = pd.DataFrame()

for _, row in unique_combinations.iterrows():
    print(_)
    date = row['Time Over'].strftime('%Y-%m-%d')
    hour = row['Time Over']
    lat = row['Latitude']
    lon = row['Longitude'] 
    
    # print(date,hour,lat,lon)
    
    weather_data = get_weather(date, lat, lon)
    hourly = weather_data.Hourly()
    
    hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
     	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
     	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
     	freq = pd.Timedelta(seconds = hourly.Interval()),
     	inclusive = "left"
    )}
    hourly_data['Latitude'] = row['Latitude']
    hourly_data['Longitude'] = row['Longitude']
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
    hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
    
    # print(hourly_data)    
    
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    
    time_diffs = (hourly_dataframe['date'] - hour).abs()
    closest_idx = time_diffs.idxmin()
    closest_row = hourly_dataframe.loc[closest_idx]
    matched_rows = pd.concat([matched_rows, closest_row.to_frame().T], ignore_index=True)

matched_rows = matched_rows.drop(columns=['date'])

X = pd.concat([X, matched_rows[['precipitation','cloud_cover','wind_speed_100m','wind_direction_100m']]], axis=1)
X["precipitation"] = X["precipitation"].astype(float)
X["cloud_cover"] = X["cloud_cover"].astype(float)
X["wind_speed_100m"] = X["wind_speed_100m"].astype(float)
X["wind_direction_100m"] = X["wind_direction_100m"].astype(float)

X = X.drop(['Time Over','Hour','Time Over','ECTRL ID'], axis=1)

X.to_csv("training.csv")  
y.to_csv('class.csv')
