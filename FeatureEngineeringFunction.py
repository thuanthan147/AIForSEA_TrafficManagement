import geohash
import pickle

SECOND_IN_HOUR = 3600
SECOND_IN_MINUTE = 60
MAX_HOUR_IN_DAY = 23
MAX_TIME_INDEX_LENGTH = len(str(MAX_HOUR_IN_DAY * SECOND_IN_HOUR)) 

# Loading 3 mapper 
with open("day_density_over_geohash.map", "rb") as f:
    DAY_DENSITY_OVER_GEO = pickle.load(f)

with open ("day_density_over_timestamp.map", "rb") as f:
    DAY_DENSITY_OVER_TIMESTAMP = pickle.load(f)

with open ("day_density_over_hour.map", "rb") as f:
    DAY_DENSITY_OVER_HOUR = pickle.load(f)


# This section is for handling single feature
## geohash6 feature
## Extract x and y coordinate from geohash code
def extract_x_coordinate(geohash_value):
    return geohash.decode(geohash_value)[0]

def extract_y_coordinate(geohash_value):
    return geohash.decode(geohash_value)[1]

## timestamp feature
## Extract hour to feature
def timestamp_to_hour(timestamp):
    time_split = timestamp.split(":")
    hour = int(time_split[0])
    return int(hour)

## timestamp to second integer
def timestamp_to_second(timestamp):
    time_split = timestamp.split(":")
    hour = int(time_split[0])
    minute = int(time_split[1])
    second = hour * SECOND_IN_HOUR + minute * SECOND_IN_MINUTE
    return second 

## Count of previous day ...
## ... for geohash6
def add_previous_day_count(input_row):
    current_day = input_row['day']
    location = input_row['geohash6']

    if current_day == 1:
        return 0
    else:
        if location not in DAY_DENSITY_OVER_GEO[current_day].keys():
            return 0
        else:
            return DAY_DENSITY_OVER_GEO[current_day][location]

## ... timestamp
def add_previous_day_count_timestamp(input_row):
    current_day = input_row['day']
    timestamp = input_row['timestamp']

    if current_day == 1:
        return 0
    else:
        if timestamp not in DAY_DENSITY_OVER_TIMESTAMP[current_day].keys():
            return 0
        else:
            return DAY_DENSITY_OVER_TIMESTAMP[current_day][timestamp]

## ... hour
## ... timestamp
def add_previous_day_count_hour(input_row):
    current_day = input_row['day']
    hour = input_row['hour']

    if current_day == 1:
        return 0
    else:
        if hour not in DAY_DENSITY_OVER_HOUR[current_day].keys():
            return 0
        else:
            return DAY_DENSITY_OVER_HOUR[current_day][hour]

## ... add seasonal patter

def add_seasonal_pattern(day):
    return day + 1 % 7


## Normalize timestamp
#  Turn day and hour to real time index
def day_hour_to_index(input_row):
    current_day = str(input_row['day'])
    hour = int(input_row['hour'])
    hour_to_sec = hour * SECOND_IN_HOUR
    zero_to_put = MAX_TIME_INDEX_LENGTH - len(str(hour_to_sec))
    return current_day + zero_to_put * "0" + str(hour_to_sec)
