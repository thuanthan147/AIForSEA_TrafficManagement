import pandas as pd
import os
from urllib import request
import zipfile

class DataHandler():
    def __init__(self):
        self.file_name = ""
        self.file_path = ""
        self.folder_path = ""

    def load_dataset(self, input_path):
        """
        Read dataset and transform it to pandas' dataframe
        Parametes:
        Input:
            input_path: string
                Location of the dataset
        Output:
            dataframe: pandas' dataframe
                Return the dataframe of input dataset
        """
        
        # Loading CSV file to pandas dataframe
        dataframe = pd.read_csv(input_path)
        return dataframe

    def download_dataset(self, dataset_url, dataset_path):
        """
        Given a url, download and save dataset to the path defined
        Input:
            dataset_url: string
                The location of the dataset on the Internet
            dataset_path: string
                Where to save the dataset
        Output:
            None
        """
        # Check if the path exists, if not, create the directory
        print("Checking if path exist")
        if not os.path.exists(dataset_path):
            print("Creating path...")
            os.makedirs(dataset_path, exist_ok=True)
            print("Created!")

        # This is the file name
        file_name = dataset_url.split('/')[-1]
        full_path = dataset_path + file_name
        
        self.file_name = file_name
        self.file_path = full_path
        self.folder_path = dataset_path

        # Check if file exists
        if os.path.exists(full_path):
            print("File has been downloaded!")
        else:
        # Download the dataset and save to the created path
            print("Downloading dataset!")
            request.urlretrieve(dataset_url, full_path)
            print("Check output file at {}!".format(full_path))

    def extract_file(self, file_name, file_path, folder_path):
        """
            This function is for extracting file (if needed)
            Only tested on .zip file
        """
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(folder_path)
        extracted_files = os.listdir(folder_path)

        extracted_files.remove(file_name)
        print("Done! Here are the extracted files/ folders")
        for file_name in extracted_files:
            print(file_name)