import pandas as pd
import pickle

def generate_mapper_for_feature(df, column_name, mode, output_file_name=None):
    """
    Function to create count mapper to increase speed
    of feature generation
    Two mode: demand or count

    @param df: pandas.DataFrame
        input dataframe
    @param column_name: string
        column to generate mapper
    @param output_file_name: string
        file to dump data
    @return None
    """
    if column_name not in df.columns:
        print("No such column in the dataframe!")
        return
    result = {}
    if mode == "demand":
        current_group = df.groupby(column_name)
        for group in current_group:
            result[group[0]] = group[1]['demand'].sum()
    
    elif mode == "count":
        group_by_day = df.groupby("day")
        list_of_day = group_by_day.groups.keys()
        for day in list_of_day:
            current_day = group_by_day.get_group(day)
            count_by_column = dict(current_day[column_name].value_counts())
            result[day] = count_by_column
    
    if output_file_name != None:
        with open(output_file_name, "wb") as f:
            pickle.dump(result, f)
    
    return result