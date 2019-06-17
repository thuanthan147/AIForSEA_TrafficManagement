import geohash
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

SECOND_IN_HOUR = 3600
SECOND_IN_MINUTE = 60
MAX_HOUR_IN_DAY = 23
MAX_TIME_INDEX_LENGTH = len(str(MAX_HOUR_IN_DAY * SECOND_IN_HOUR)) 
DATABASE_PATH = "database/"

# Loading mapper 
with open(DATABASE_PATH + "day_density_over_geohash.map", "rb") as f:
    DAY_DENSITY_OVER_GEO = pickle.load(f)

with open(DATABASE_PATH + "day_density_over_timestamp.map", "rb") as f:
    DAY_DENSITY_OVER_TIMESTAMP = pickle.load(f)

with open(DATABASE_PATH + "day_density_over_hour.map", "rb") as f:
    DAY_DENSITY_OVER_HOUR = pickle.load(f)

with open(DATABASE_PATH + "geo_density_hour.map", "rb") as f:
    GEO_DENSITY_HOUR = pickle.load(f)

with open(DATABASE_PATH + "geo_density_day.map", "rb") as f:
    GEO_DENSITY_DAY = pickle.load(f)

with open(DATABASE_PATH + "geo_to_density.map", "rb") as f:
    GEO_TO_DENSITY = pickle.load(f)    



def add_density(df):
    """
    Add label density (low, medium, high) for each geohash6 value
    This feature is generated through a pretrained KMeans
    Args:
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.DataFrame: The datatframe after group_geohash hash been added
    """
    temp_df = df
    geohash6_value = temp_df['geohash6'].values
    geohash_density = [GEO_TO_DENSITY[i] for i in geohash6_value]
    temp_df['density'] = geohash_density
    return temp_df


def add_group_geohash(df):
    """
    Add group geohash, which is basically the first 4 letter of geohash6
    
    Args:
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.DataFrame: The datatframe after group_geohash hash been added
    """
    temp_df = df
    geohash6_value =temp_df['geohash6'].values
    geohash_group = [i[:4] for i in geohash6_value]
    temp_df['group_geohash'] = geohash_group
    
    return temp_df

def add_new_axis(df):
    """
    Explode the geohash6 feature to x and y axis

    Args:
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.DataFrame: The dataframe after add x and y axis
    """
    temp_df = df
    geohash6_value = temp_df['geohash6'].values
    coordinates = [geohash.decode(i) for i in geohash6_value]

    temp_df['x'] =  [i[0] for i in coordinates]
    temp_df['y'] =  [i[1] for i in coordinates]
    return temp_df


def add_minute(df):
    """
    Create minute feature from timestamp
    Include a closure function with split the timestamp and return the second element

    Args:
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.Dataframe: The dataframe after add minute
    """
    def timestamp_to_minute(timestamp):
        time_split = timestamp.split(":")
        minute = int(time_split[1])
        return minute
    temp_df = df
    timestamp_value = temp_df['timestamp'].values
    minute = [timestamp_to_minute(timestamp) for timestamp in timestamp_value]
    temp_df['minute'] =  minute
    return temp_df


def add_hour(df):
    """
    Create hour feature from timestamp
    Include a closure function with split the timestamp and return the first element

    Args:
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.Dataframe: The dataframe after add hour count
    """
    def timestamp_to_hour(timestamp):
        time_split = timestamp.split(":")
        hour = int(time_split[0])
        return int(hour)
    temp_df = df
    timestamp_value = temp_df['timestamp'].values
    hour_value = [timestamp_to_hour(timestamp) for timestamp in timestamp_value]
    temp_df['hour'] =  hour_value
    return temp_df
    

def add_previous_count_geo(df, lag_num=1, update_mapping=False):
    """
    Get the previous day total appearance of the current geohash6
    Args:
        df (pandas.DataFrame): The dataframe to add
        lag_num (int): The lag value (t - lag_num  day before), default 1
        update_mapping (Boolean): whether or not to update mapper
    Returns:
        pandas.DataFrame: the dataframe after add geo count
    """
    def extract_previous_day_count(current_day, location, lag_num):
        """
        Using the mapper to get the count of current location at lag_num day before
        Args:
            current_day (int): The current day
            location (str): The current location
            lag_num (int): The lag value, default 1
        Returns:
            result (int): total appearance of location previous day
        """
        day_index = current_day - lag_num
        result = 0
        if day_index <= 0:
            return 0
        else:
            if day_index not in DAY_DENSITY_OVER_GEO.keys():
                return result
            if location not in DAY_DENSITY_OVER_GEO[day_index].keys():
                return result
            else:
                result = DAY_DENSITY_OVER_GEO[day_index][location] 
                return result

    temp_df = df
    day_and_geohash = df[['day', 'geohash6']].values
    day_index_count = [extract_previous_day_count(i[0], i[1], lag_num) for i in day_and_geohash]
    temp_df['day_over_geo_' + str(lag_num)] = day_index_count
                                                
    return temp_df
    
def add_previous_count_timestamp(df, lag_num=1, update_mapping=False):
    """
    Get the previous day total appearance of the current timestamp
    Args:
        df (pandas.DataFrame): The dataframe to add
        lag_num (int): The lag value (t - lag_num  day before), default 1
        update_mapping (Boolean): whether or not to update mapper
    Returns:
        pandas.DataFrame: the dataframe after add timestamp count
    """
    def extract_previous_day_count_timestamp(current_day, timestamp, lag_num):
        """
        Using the mapper to get the count of current timestamp at lag_num day before
        Args:
            current_day (int): The current day
            location (str): The current location
            lag_num (int): The lag value, default 1
        Returns:
            result (int): total appearance of timestamp previous day
        """
        day_index = current_day - lag_num
        
        result = 0
        if day_index <= 0:
            return result
        else:
            if day_index not in DAY_DENSITY_OVER_TIMESTAMP.keys():
                return result
            if timestamp not in DAY_DENSITY_OVER_TIMESTAMP[day_index].keys():
                return result
            else:
                result = DAY_DENSITY_OVER_TIMESTAMP[day_index][timestamp] 
                return result

    temp_df = df
    day_and_timestamp = temp_df[['day', 'timestamp']].values
    day_index_count = [extract_previous_day_count_timestamp(i[0], i[1], lag_num) for i in day_and_timestamp]
    
    temp_df['day_over_timestamp_' + str(lag_num)] = day_index_count

    return temp_df
    

def add_previous_count_hour(df, lag_num=1, update_mapping=False):
    """
    Get the previous day total appearance of the current hour
    Check whether there is an hour feature before continue
    Args:
        df (pandas.DataFrame): The dataframe to add
        lag_num (int): The lag value (t - lag_num  day before), default 1
        update_mapping (Boolean): whether or not to update mapper
    Returns:
        pandas.DataFrame: the dataframe after add hour count
    """
    def extract_previous_day_count_hour(current_day, hour, lag_num):
        """
        Using the mapper to get the count of current hour at lag_num day before
        Args:
            current_day (int): The current day
            location (str): The current location
            lag_num (int): The lag value, default 1
        Returns:
            result (int): total appearance of hour previous day
        """
        day_index = current_day - lag_num
        result = 0
        if day_index <= 0:
            return result
        else:
            if day_index not in DAY_DENSITY_OVER_HOUR.keys():
                return result
            if hour not in DAY_DENSITY_OVER_HOUR[day_index].keys():
                return result
            else:
                result = DAY_DENSITY_OVER_HOUR[day_index][hour] 
                return result

    temp_df = df
    if 'hour' not in temp_df.columns:
        temp_df = add_hour(temp_df)
    
    day_and_hour = df[['day', 'hour']].values
    day_index_count = [extract_previous_day_count_hour(i[0], i[1], lag_num) for i in day_and_hour]

    temp_df['day_over_hour_' + str(lag_num) ] = day_index_count
    return temp_df
    

def add_day_of_week_pattern(df):
    """
    Add day of week (after visualize and get the seasonal pattern)
    Args:
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.Dataframe: The dataframe after generated day_of_week feature
    """
    def extract_seasonal_pattern(day):
        return (int(day) + 1) % 7

    temp_df = df
    day_value = temp_df['day'].values
    seasonal_pattern = [extract_seasonal_pattern(i) for i in day_value]
    temp_df['day_of_week'] = seasonal_pattern
    return temp_df
    

def add_time_index(df):
    """
    Add time index feature which is the combination of day and timestamp (transformed to milisecond)
    Args: 
        df (pandas.DataFrame): The dataframe to add
    Returns:
        pandas.DataFrame: The dataframe with time index feature
    """
    def day_hour_to_index(input_day, input_hour):
        current_day = str(input_day)
        hour = int(input_hour)
        hour_to_sec = hour * SECOND_IN_HOUR
        zero_to_put = MAX_TIME_INDEX_LENGTH - len(str(hour_to_sec))
        return current_day + zero_to_put * "0" + str(hour_to_sec)
    
    temp_df = df
    if 'hour' not in temp_df.columns:
        temp_df = add_hour(temp_df)
    day_and_hour = temp_df[['day', 'hour']].values
    time_index = [day_hour_to_index(i[0], i[1]) for i in day_and_hour]
    temp_df['time_index'] = time_index
                                            
    return temp_df
    
def add_statistic_features(df, raw_feature_name, start_time, time_range, specific=None):
    temp_df = df
    features = []
    end_time = start_time + time_range
    for i in range(start_time, end_time + 1):
        features.append(raw_feature_name + "_" + str(i))
    current_feature = temp_df[features].values
    
    means = [i.mean() for i in current_feature]
    max_val = [i.max() for i in current_feature]
    min_val = [i.min() for i in current_feature]

    mean_name = raw_feature_name + "_mean_w_" + str(start_time) + "_" + str(end_time)
    min_name = raw_feature_name + "_min_w_" + str(start_time) + "_" + str(end_time)
    max_name = raw_feature_name + "_max_w_" + str(start_time) + "_" + str(end_time)

    if specific == None:
        temp_df[mean_name] = means
        temp_df[min_name] = min_val
        temp_df[max_name] = max_val
    elif specific == "mean":
        temp_df[mean_name] = means
    elif specific == "max":
        temp_df[max_name] = max_val
    elif specific == "min":
        temp_df[min_name] = min_val
    else:
        print("No suitable method found!")
    return temp_df


def dummyEncode(df, cols):
    temp_df = df
    le = LabelEncoder()
    for feature in cols:
        try:
            temp_df[feature] = le.fit_transform(temp_df[feature])
        except:
            print('Error encoding '+feature)
    return temp_df


def one_hot(df):
    """
    @param temp_df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    temp_df = df
    cols = list(temp_df.select_dtypes(include=['category','object']))

    for each in cols:
        dummies = pd.get_dummies(temp_df[each], prefix=each, drop_first=False)
        temp_df = pd.concat([temp_df, dummies], axis=1)
    return temp_df


def add_all_features(df):
    temp_df = df
    temp_df = add_previous_count_hour(temp_df)
    temp_df = add_previous_count_geo(temp_df)
    temp_df = add_previous_count_timestamp(temp_df)
    temp_df = add_day_of_week_pattern(temp_df)
    temp_df = add_time_index(temp_df)
    temp_df = add_new_axis(temp_df)
    return temp_df

# # Remove over-generated lag features 
# import pandas as pd
# temp_df = pd.DataFrame() # holder dataframe
# columns_name = temp_df.columns
# look_out = ['day_over_hour', 'day_over_timestamp', 'day_over_geo']
# keep_columns = []

# for name in columns_name:
#     if name == "Unnamed: 0" or name == 'x' or name == 'y':
#         continue
#     if any(sub_name in name for sub_name in look_out):
#         final_num = int(name.split("_")[-1])
#         if final_num < 4:
#             keep_columns.append(name)
#         continue
#     keep_columns.append(name)

# temp_df = temp_df[keep_columns]



def add_previous_hour_demand_by_geo(df, lag_hour=1, update_mapping=False):
    """
    Add total demand for lag_hour before for current geo
    Args:
        df (pandas.DataFrame): The dataframe to add
        lag_hour (int): how many hours we want to go back
        update_mapping (bool): whether or not to update mapping for current function
                                (default False)
    Returns:
        pandas.DataFrame: The dataframe after added the hour demand
    """
    def extract_previous_hour_demand(current_time_index, location, lag_num):
        """
        Using the mapper to get the count of current hour at lag_num day before
        Args:
            current_time_index (int): The current time index
            location (str): The current location
            lag_num (int): The lag value, default 1
        Returns:
            result (int): total appearance of hour previous day
        """
        # print(current_time_index)
        if int(current_time_index) % 100000 == 0:
            head = int(current_time_index) / 100000 - 1
            if head <= 0: 
                time_index = -1
            else:
                time_index = str(int(head)) + str(23 * SECOND_IN_HOUR)
        else :
            time_index = int(current_time_index) - lag_num * SECOND_IN_HOUR
        result = 0
        if int(time_index) <= 0:
            return result
        else:
            if str(time_index) not in GEO_DENSITY_HOUR.keys():
                return result
            if location not in GEO_DENSITY_HOUR[str(time_index)].keys():
                return result
            else:
                result = GEO_DENSITY_HOUR[str(time_index)][location] 
                return result

    temp_df = df

    if 'time_index' not in temp_df.columns:
        temp_df = add_time_index(temp_df)

    if update_mapping == True:
        index_to_demand = {}
        group_of_time_index = temp_df.groupby('time_index')
        for group in group_of_time_index:
            geo_to_demand = {}
            group_of_geo = group[1].groupby('geohash6')
            for geo_group in group_of_geo:
                geo_to_demand[geo_group[0]] = geo_group[1]['demand'].sum()
            index_to_demand[group[0]] = geo_to_demand
        
        with open(DATABASE_PATH + "geo_density_hour.map", "wb") as f:
            pickle.dump(index_to_demand, f)
    temp_df = df
    day_and_hour = temp_df[['time_index', 'geohash6']].values
    time_index = [extract_previous_hour_demand(i[0], i[1], lag_hour) for i in day_and_hour]
    temp_df['geo_demand_hour_' + str(lag_hour)] = time_index
    return temp_df


def add_previous_day_demand_by_geo(df, lag_day=1, update_mapping=False):
    """
    Add total demand for lag_day before for current geo
    Args:
        df (pandas.DataFrame): The dataframe to add
        lay_day (int): how many days we want to go back
        update_mapping (bool): whether or not to update mapping for current function
                                (default False)
    Returns:
        pandas.DataFrame: The dataframe after added the day demand
    """
    def extract_previous_day_demand(current_day, location, lag_day):
        """
        Using the mapper to get the count of current demand at lag_num day before
        Args:
            current_day (int): The current day
            location (str): The current location
            lag_day (int): The lag value, default 1
        Returns:
            result (int): total demand of lag_day previous day
        """
        day_index = current_day - lag_day
        result = 0
        if day_index <= 0:
            return result
        else:
            if day_index not in GEO_DENSITY_DAY.keys():
                return result
            if location not in GEO_DENSITY_DAY[day_index].keys():
                return result
            else:
                result = GEO_DENSITY_DAY[day_index][location] 
                return result

    temp_df = df

    if update_mapping == True:
        day_to_demand = {}
        group_of_day = temp_df.groupby('day')
        for group in group_of_day:
            geo_to_demand = {}
            group_of_geo = group[1].groupby('geohash6')
            for geo_group in group_of_geo:
                geo_to_demand[geo_group[0]] = geo_group[1]['demand'].sum()
            day_to_demand[group[0]] = geo_to_demand
        
        with open(DATABASE_PATH + "geo_density_day.map", "wb") as f:
            pickle.dump(day_to_demand, f)
    temp_df = df
    day_and_hour = temp_df[['day', 'geohash6']].values
    time_index = [extract_previous_day_demand(i[0], i[1], lag_day) for i in day_and_hour]
    temp_df['geo_demand_day_' + str(lag_day)] = time_index
    return temp_df