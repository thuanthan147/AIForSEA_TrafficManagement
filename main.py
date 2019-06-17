from config import DATASET_PATH, DATASET_URL
from data_handling import DataHandler
from model.feature_engineering.FeatureEngineeringFunction import add_hour
from model.feature_engineering.FeatureEngineeringFunction import add_minute
from model.feature_engineering.FeatureEngineeringFunction import add_group_geohash
from model.feature_engineering.FeatureEngineeringFunction import add_previous_count_geo
from model.feature_engineering.FeatureEngineeringFunction import add_statistic_features
from model.feature_engineering.FeatureEngineeringFunction import add_previous_count_hour
from model.feature_engineering.FeatureEngineeringFunction import add_previous_count_timestamp
from model.feature_engineering.FeatureEngineeringFunction import add_previous_day_demand_by_geo
from model.feature_engineering.FeatureEngineeringFunction import add_previous_hour_demand_by_geo
from model.feature_engineering.FeatureEngineeringFunction import add_density
from model.feature_engineering.FeatureEngineeringFunction import add_time_index
from model.feature_engineering.FeatureEngineeringFunction import dummyEncode

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from mlxtend.regressor import StackingRegressor
from lightgbm import LGBMRegressor
import numpy as np
import pickle
import pandas as pd
import argparse

DATABASE_PATH = "database/"

def load_dataframe():
    data_handler = DataHandler()
    data_handler.download_dataset(DATASET_URL, DATASET_PATH)
    data_handler.extract_file(data_handler.file_name, 
                          data_handler.file_path, 
                          data_handler.folder_path)
    FULL_PATH = DATASET_PATH + "Traffic Management/training.csv"
    
    print(FULL_PATH)
    
    return data_handler.load_dataset(FULL_PATH)


def feature_engineering(df):
    temp_df = df

    temp_df = add_hour(temp_df)
    temp_df = add_time_index(temp_df)
    temp_df = add_group_geohash(temp_df)

    # Count by geo feature
    temp_df = add_previous_count_geo(temp_df, lag_num=1)
    # Holder feature
    temp_df = add_previous_count_geo(temp_df, lag_num=2)
    temp_df = add_previous_count_geo(temp_df, lag_num=3)
    temp_df = add_previous_count_geo(temp_df, lag_num=4)
    temp_df = add_previous_count_geo(temp_df, lag_num=5)

    temp_df = add_statistic_features(temp_df, "day_over_geo", 1, 4, specific="mean")
    temp_df = add_statistic_features(temp_df, "day_over_geo", 1, 4, specific="min")

    # Drop it after generated statistic
    temp_df = temp_df.drop(columns=['day_over_geo_2', 'day_over_geo_3', 'day_over_geo_4', 'day_over_geo_5'])

    temp_df = add_previous_count_geo(temp_df, lag_num=6)
    temp_df = add_previous_count_geo(temp_df, lag_num=7)

    # Count by hour feature
    temp_df = add_previous_count_hour(temp_df, lag_num=1)
    # Holder features
    temp_df = add_previous_count_hour(temp_df, lag_num=2)
    temp_df = add_previous_count_hour(temp_df, lag_num=3)
    temp_df = add_previous_count_hour(temp_df, lag_num=4)
    temp_df = add_previous_count_hour(temp_df, lag_num=5)

    temp_df = add_statistic_features(temp_df, "day_over_hour", 1, 4, specific="min")
    temp_df = add_statistic_features(temp_df, "day_over_hour", 1, 4, specific="max")

    # Drop it after generated statistic
    temp_df = temp_df.drop(columns=['day_over_hour_2', 'day_over_hour_3', 'day_over_hour_4', 'day_over_hour_5'])

    temp_df = add_previous_count_hour(temp_df, lag_num=6)
    temp_df = add_previous_count_hour(temp_df, lag_num=7)

    # Count by timestamp feature
    temp_df = add_previous_count_timestamp(temp_df, lag_num=1)
    temp_df = add_previous_count_timestamp(temp_df, lag_num=3)

    # Holder features
    temp_df = add_previous_count_timestamp(temp_df, lag_num=2)
    temp_df = add_previous_count_timestamp(temp_df, lag_num=4)
    temp_df = add_previous_count_timestamp(temp_df, lag_num=5)
    temp_df = add_statistic_features(temp_df, "day_over_timestamp", 1, 4, specific="min")
    
    # Drop it after generate statistic
    temp_df = temp_df.drop(columns=['day_over_timestamp_2', 'day_over_timestamp_4', 'day_over_timestamp_5'])

    temp_df = add_previous_count_timestamp(temp_df, lag_num=6)
    temp_df = add_previous_count_timestamp(temp_df, lag_num=7)
    
    
    temp_df = add_previous_hour_demand_by_geo(temp_df, lag_hour=1)
    temp_df = add_previous_hour_demand_by_geo(temp_df, lag_hour=2)
    temp_df = add_previous_hour_demand_by_geo(temp_df, lag_hour=3)
    temp_df = add_previous_hour_demand_by_geo(temp_df, lag_hour=4)
    temp_df = add_previous_hour_demand_by_geo(temp_df, lag_hour=5)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=1)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=2)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=3)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=4)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=5)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=6)
    temp_df = add_previous_day_demand_by_geo(temp_df, lag_day=7)
    temp_df = add_minute(temp_df)
    temp_df = add_density(temp_df)

    return temp_df


def get_rmse_score(result, y):
    return np.sqrt(mean_squared_error(result, y))

def train_model(X_train, y_train):
    clf1 = LinearSVR()
    clf2 = LinearRegression()
    clf3 = Ridge()
    clf4 = LGBMRegressor()

    svr_linear = LinearSVR()
    sr = StackingRegressor(regressors=[clf1, clf2, clf3, clf4], meta_regressor=svr_linear)
    
    sr.fit(X_train, y_train)
    result = sr.predict(X_train)
    score = get_rmse_score(result, y_train)
    print("RMSE Score train: %.4f" % score)
    return sr


def normalize_dataframe(df):
    temp_df = df
    # temp_df = temp_df.drop(columns=['day'])

    with open("min_max_scaler.pkl", "rb") as f:
        min_max_scaler = pickle.load(f)
    with open("standard_scaler.pkl", "rb") as f:
        standard_scaler = pickle.load(f)

    categorical_columns = ['geohash6', 'group_geohash', 'hour', 'minute']
    minmax = temp_df[categorical_columns]
    scaled_values = min_max_scaler.fit_transform(minmax) 
    minmax.loc[:,:] = scaled_values
    
    zscore = temp_df.drop(columns=categorical_columns)
    scaled_values = standard_scaler.fit_transform(zscore)
    zscore.loc[:,:] = scaled_values
    
    final_df = pd.concat([minmax, zscore], axis = 1)
    
    return final_df.values


def normalize_dataframe_first_run(df):
    temp_df = df

    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    categorical_columns = ['geohash6', 'group_geohash', 'hour', 'minute']
    minmax = temp_df[categorical_columns]
    scaled_values = min_max_scaler.fit_transform(minmax) 
    minmax.loc[:,:] = scaled_values
    
    zscore = temp_df.drop(columns=categorical_columns)
    scaled_values = standard_scaler.fit_transform(zscore)
    zscore.loc[:,:] = scaled_values
    
    final_df = pd.concat([minmax, zscore], axis = 1)
    
    return final_df



if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', help='The T to T+13')
    parser.add_argument('--test_set', help='Perform predict on this set')
    args = parser.parse_args()
    

    train_path = args.train_set 
    test_path = args.test_set

    print("Reading data!")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = add_hour(train_df)
    train_df = add_time_index(train_df)


    day_density_over_geo = {}
    day_density_over_timestamp = {}
    day_density_over_hour = {}
    geo_density_hour = {}
    geo_density_day = {}

    print("Generate mapping!")
    group_by_day = train_df.groupby("day")
    print(set(train_df['day'].values.tolist()))
    for group in group_by_day:
        current_day = group[0]

        geo_count = {}
        group_geo = group[1].groupby("geohash6")
        for geo in group_geo:
            current_geo = geo[0]
            total = len(geo[1])
            geo_count[current_geo] = total
        day_density_over_geo[current_day] = geo_count

        timestamp_count = {}
        group_timestamp = group[1].groupby("timestamp")
        for timestamp in group_timestamp:
            current_timestamp = geo[0]
            total = len(timestamp[1])
            timestamp_count[current_timestamp] = total
        day_density_over_timestamp[current_day] = timestamp_count

        hour_count = {}
        group_hour = group[1].groupby("hour")
        for hour in group_hour:
            current_hour = hour[0]
            total = len(hour[1])
            hour_count[current_hour] = total
        day_density_over_hour[current_day] = hour_count

        geo_demand = {}
        group_geo = group[1].groupby("geohash6")
        for geo in group_geo:
            current_geo = geo[0]
            total_demand = geo[1]['demand'].sum()
            geo_demand[current_geo] = total_demand
        geo_density_day[current_day] = geo_demand
    print(geo_density_day)

    print(train_df.columns)
    group_by_time_index = train_df.groupby("time_index")
    for group in group_by_time_index:
        geo_demand = {}
        current_index = group[0]
        group_geo = group[1].groupby("geohash6")
        for geo in group_geo:
            current_geo = geo[0]
            total_demand = geo[1]['demand'].sum()
            geo_demand[current_geo] = total_demand
        geo_density_hour[current_index] = geo_demand 
    print(geo_density_hour)
    train_df = train_df.drop(columns=['Unnamed: 0'])



    day_density_over_geo = {}
    day_density_over_timestamp = {}
    day_density_over_hour = {}
    geo_density_hour = {}
    geo_density_day = {}

    print("Saving mapper!")
    with open(DATABASE_PATH + "day_density_over_geohash.map", "wb") as f:
        pickle.dump(day_density_over_geo, f)

    with open(DATABASE_PATH + "day_density_over_timestamp.map", "wb") as f:
        pickle.dump(day_density_over_timestamp, f)

    with open(DATABASE_PATH + "day_density_over_hour.map", "wb") as f:
        pickle.dump(day_density_over_hour, f)

    with open(DATABASE_PATH + "geo_density_hour.map", "wb") as f:
        pickle.dump(geo_density_hour, f)

    with open(DATABASE_PATH + "geo_density_day.map", "wb") as f:
        pickle.dump(geo_density_day, f)


    print("Loading model!")
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    list_of_day = list(set(train_df['day'].values.tolist()))
    X = add_time_index(test_df)
    X = add_hour(X)
    X = X.sort_values(['day', 'hour'], ascending=[True, True])
    print(X['day'])

    print("Feature engineering and predicting!")

    X['dummy_day'] = X['day']
    X['dummy_time_index'] = X['time_index']
    group_to_train = X.groupby('dummy_day')
    
    result = []
    for group in group_to_train:
        current_day = group[0]
        index_group = group[1].groupby("dummy_time_index")
        for index in index_group:
            current_data = index[1].drop(columns=["Unnamed: 0", 'dummy_day', 'dummy_time_index'])
            if "demand" in current_data.columns:
                current_data = current_data.drop(columns=["demand"])
            current_df = current_data
            current_data = feature_engineering(current_data)

            current_data = current_data.drop(columns=['timestamp'])
            current_data = current_data.fillna(0)
            encode_cols = ['geohash6', "group_geohash"]
            current_data = dummyEncode(current_data, encode_cols)
            current_data = normalize_dataframe(current_data)

            
            y = model.predict(current_data)
            result += list(y)
            current_df['demand'] = y
            if current_day not in list_of_day:
                list_of_day.append(current_day)
                holder_group = current_df.groupby("geohash6")
                holder_mapper = {}
                for g in holder_group:
                    current_geo = g[0]
                    total = len(g[1])
                    holder_mapper[current_geo] = total
                day_density_over_geo[current_day] = holder_mapper

                holder_group = current_df.groupby("timestamp")
                holder_mapper = {}
                for g in holder_group:
                    current = g[0]
                    total = len(g[1])
                    holder_mapper[current] = total
                day_density_over_timestamp[current_day] = holder_mapper

                holder_group = current_df.groupby("hour")
                holder_mapper = {}
                for g in holder_group:
                    current = g[0]
                    total = len(g[1])
                    holder_mapper[current] = total
                day_density_over_timestamp[current_day] = holder_mapper

                holder_group = current_df.groupby("geohash6")
                holder_mapper = {}
                for g in holder_group:
                    current = g[0]
                    total = g[1]['demand'].sum()
                    holder_mapper[current] = total
                geo_density_day[current_day] = holder_mapper

                holder_group = current_df.groupby("time_index")
                holder_mapper = {}
                for g in holder_group:
                    current = g[0]
                    total = g[1]['demand'].sum()
                    holder_mapper[current] = total
                geo_density_hour[current_day] = holder_mapper

    print("--------------------------------------------")
    print("Predicted results: %.4f" % np.sqrt(mean_squared_error(result, X['demand'].values)))

    print("Writing result to file")
    with open("output/result.txt", "w") as f:
        for i in result:
            f.write(str(i))
            f.write("\n")
    # day_density_over_geo = {}
    # day_density_over_timestamp = {}
    # day_density_over_hour = {}
    # geo_density_hour = {}
    # geo_density_day = {}
