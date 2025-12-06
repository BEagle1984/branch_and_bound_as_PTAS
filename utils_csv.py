import pandas as pd
import os


# folder where csv files are located
CSV_FOLDER = "output/"

# list of csv filenames to merge:
CSV_FILENAMES = [
    "results-sergio-e0.05.csv",
    "results-tiziano-e0.05.csv"
]

# output merged csv filename
MERGED_CSV_FILENAME = "results-e0.05.csv"

# load all csv data into a single dataframe
def load_csv_data(csv_folder=CSV_FOLDER, csv_filenames=CSV_FILENAMES, csv_delimiter=',', csv_header=0) -> pd.DataFrame:    
    dataframes = []
    for filename in csv_filenames:
        filepath = os.path.join(csv_folder, filename)
        df = pd.read_csv(filepath, delimiter=csv_delimiter, header=csv_header)
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


# display dataframe head and tail
df = load_csv_data()
print("Dataframe Head:")
print(df.head())
print("\nDataframe Tail:")
print(df.tail())

# save merged dataframe to a new csv file
output_filepath = os.path.join(CSV_FOLDER, MERGED_CSV_FILENAME)
df.to_csv(output_filepath, index=False)
print(f"\nMerged dataframe saved to {output_filepath}")