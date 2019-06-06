import geohash

SECOND_IN_HOUR = 3600
SECOND_IN_MINUTE = 60

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