import pandas as pd
import os
import numpy as np

def merge_all_files(input_directory, output_file_path):
    """
    There are 42 different traffic signs. Each signs have about 150 pictures each. Every picture has feature vectors extracted in directory HOG_01.
    All the feature vectors are just vectors and random forest needs a feature matrix. This function gets all the feature vectors from each traffic sign
    and merges them into one file so output for this is a directory of 42 txt files that contains feature vectors for each traffic sign. After this
    transpose_and_merge_files() function merges all the 42 files to one so it can be an output to random forest. 

    :param input_directory: Directory containing files to be merged.
    :param output_file_path: Path to save the merged file.
    """
    all_dfs = [] 

    for file in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file)
        
        # Check if it's a file
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, header=None)  # Read the file into a DataFrame
            all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    merged_df.to_csv(output_file_path, index=False, header=False)

def GetAllLabels(input_directory, output_file_path):
    """
    This function extracts labels from directory Dataset/Training where every traffic sign has its own csv file which contains a classID column.
    
    """
    labelsArray = []
    csv_file_paths = []

    for file in os.listdir(input_directory):
        # check if it is a directory
        if os.path.isdir(os.path.join(input_directory, file)):
            # enter the directory and find csv file in it
            subdirectory_path = os.path.join(input_directory, file)
            for subfile in os.listdir(subdirectory_path):
                if subfile.endswith('.csv'):
                    csv_file_path = os.path.join(subdirectory_path, subfile)
                    csv_file_paths.append(csv_file_path)
    csv_file_paths.sort()
    for path in csv_file_paths:
        with open(path, 'r') as file:
            labels = pd.read_csv(path, sep=';')
        labelsArray.append(labels.iloc[:, -1].values)

    all_labels = np.concatenate(labelsArray)
    # save the all_labels to a text file
    np.savetxt(output_file_path, all_labels, fmt='%s')

#Merges all the feature vectors from each traffic sign into one file.
def transpose_and_merge_files(parent_directory, output_directory):    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each subdirectory in the parent directory
    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)
        
        if os.path.isdir(subdir_path):
            transposed_dfs = []

            for file in os.listdir(subdir_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(subdir_path, file)
                    df = pd.read_csv(file_path, sep="\t", header=None)
                    transposed_df = df.transpose()  # Transpose the DataFrame
                    transposed_dfs.append(transposed_df)

            merged_df = pd.concat(transposed_dfs, ignore_index=True)

            output_file_path = os.path.join(output_directory, f'merged_vector_{subdir}.txt')
            merged_df.to_csv(output_file_path, index=False, header=False)

#parent_directory = 'Dataset/HOG_01/'
#output_directory = 'merged_output'  # Directory where merged files will be saved
#transpose_and_merge_files(parent_directory, output_directory)


# merges all the features in to one file
#input_directory = 'merged_output'
#output_file_path = 'final_merged_file.txt'  # Update with your desired output path
#merge_all_files(input_directory, output_file_path)

#input_directory = 'Dataset/Training/'
#GetAllLabels(input_directory, 'labels.txt')