import os
import time
import logging
import pandas as pd
import numpy as np

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/cardiomicscore/data/processed/covariates/'
log_dir = '/your path/cardiomicscore/saved/log/DataPreparation/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Set up logger
log_filename = os.path.join(log_dir, 'Demographic.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)

target_FieldID = [
    '54-0.0', # assessment center
    '21022-0.0', # age at recruitment
    '31-0.0', # sex
    '21000-0.0', # ethnicity
    '22189-0.0', # Townsend index
    # '53-0.0', # date of baseline assessment
]
target_FieldID = ['eid'] + target_FieldID
feature_df = population_char_df[target_FieldID]

########################### Ethnic background mapping ###############################################
# Define the mapping from the original ethnicity codes to the new simplified categories.
# New categories are: 0:White, 1:Asian, 2:Black, 3:Mixed/Others
ethnicity_mapping = {
    1: 0, 1001: 0, 1002: 0, 1003: 0,
    3: 1, 3001: 1, 3002: 1, 3003: 1, 3004: 1,
    4: 2, 4001: 2, 4002: 2, 4003: 2,
    2: 3, 2001: 3, 2002: 3, 2003: 3, 2004: 3,
    5: 3, 6: 3, -1: 3, -3: 3
}

feature_df['21000-0.0'] = feature_df['21000-0.0'].replace(ethnicity_mapping)

########################### Recode the assessment center into the region  ###############################################
feature_df['region'] = feature_df['54-0.0'].copy()
feature_df['region'].replace([10003, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
                             11011, 11012, 11013, 11014, 11016, 11017, 11018, 11020, 11021, 11022, 11023],
                            [2,     2,     7,     1,     9,     9,     5,     7,     2,     3,     4,
                             8,     0,     6,     4,     2,     3,     0,     0,     5,     1,     1], inplace = True)


# Rename columns
feature_df.rename(columns = {'21022-0.0': 'age', '31-0.0': 'male', '21000-0.0': 'ethnicity', '22189-0.0': 'townsend', '54-0.0': 'center'}, inplace=True)
feature_df = feature_df.drop(columns=['center'])
feature_df.to_csv(output_dir + 'DemographicInfo.csv', index=False)

# Calculate missing percentage
missing_df = feature_df.isnull().mean().reset_index()
missing_df.columns = ['variable', 'missing_percentage']
missing_df['missing_percentage'] *= 100

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')
logger.info("\nMissing Values Report:\n")
logger.info(missing_df.to_string(index=False))

'''

ethnicity background mapping:
'White': ["White", "British", "Irish", "Any other white background"],
'Mixed': ["Mixed", "White and Black Caribbean", "White and Black African", "White and Asian", "Any other mixed background"],  
'Asian': ["Asian or Asian British", "Indian", "Pakistani", "Bangladeshi", "Any other Asian background"], 
'Black': ["Black or Black British", "Caribbean", "African", "Any other Black background"],
'Others: ["Chinese", "Other ethnic group", "Do not know", "Prefer not to answer"]

region mapping:
'London' = 0
'Wales' = 1
'North-West' = 2
'North-East' = 3
'Yorkshire and Humber' = 4
'West Midlands' = 5
'East Midlands' = 6
'South-East' = 7
'South-West' = 8
'Scotland' = 9

'''

'''
Original region code:
https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=10

coding	meaning
11012	Barts
11021	Birmingham
11011	Bristol
11008	Bury
11003	Cardiff
11024	Cheadle (revisit)
11020	Croydon
11005	Edinburgh
11004	Glasgow
11018	Hounslow
11010	Leeds
11016	Liverpool
11001	Manchester
11017	Middlesborough
11009	Newcastle
11013	Nottingham
11002	Oxford
11007	Reading
11014	Sheffield
10003	Stockport (pilot)
11006	Stoke
11022	Swansea
11023	Wrexham
11025	Cheadle (imaging)
11026	Reading (imaging)
11027	Newcastle (imaging)
11028	Bristol (imaging)

'''