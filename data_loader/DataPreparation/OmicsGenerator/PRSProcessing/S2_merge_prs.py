# ==============================================================================
#                      Polygenic Risk Score (PRS) Merger
# ==============================================================================
#
# Description:
#   This script finds all PRS files (*.tsv) in a specified directory,
#   filters out specified diseases, merges the rest into a single table
#   using an outer join on 'eid', calculates missingness, and logs the process.
#

# --- 1. Import necessary libraries ---
import os
import time
import logging
import pandas as pd
import glob
from functools import reduce

# --- 2. Define Paths ---
# Input directory where the individual PRS .tsv files are located
data_dir = '/your path/cardiomicscore/data/processed/prs/'

# Output directory for the final combined PRS table
output_dir = '/your path/cardiomicscore/data/processed/omics/'

# Directory for the log file
log_dir = '/your path/cardiomicscore/saved/log/DataPreparation/'
os.makedirs(log_dir, exist_ok=True)

# --- 3. Set up Logger ---
log_filename = os.path.join(log_dir, 'PRSMerge.log')
# Configure logger to write to a file
logging.basicConfig(level=logging.INFO,
                    filename=log_filename,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Also configure a handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
# Clear previous handlers and add new ones to avoid duplicate logs in interactive sessions
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(logging.FileHandler(log_filename, mode='w'))
logger.addHandler(console_handler)

# --- 4. Main Processing Logic ---
logger.info("--- Starting PRS file merge process ---")
start_time = time.time()

try:
    # --- A. Find all .tsv files in the data directory ---
    all_prs_files = glob.glob(os.path.join(data_dir, '*.tsv'))
    
    if not all_prs_files:
        raise FileNotFoundError("No .tsv files found in the specified directory.")
        
    logger.info(f"Found {len(all_prs_files)} total .tsv files in the directory.")

    # --- B. Filter out the specified PRS files ---
    # Define which files to exclude from the merge
    files_to_exclude = ['aaa.tsv', 'va.tsv']
    logger.info(f"Excluding files matching: {files_to_exclude}")
    
    prs_files_to_merge = [
        f for f in all_prs_files
        if os.path.basename(f) not in files_to_exclude
    ]

    num_excluded = len(all_prs_files) - len(prs_files_to_merge)
    logger.info(f"Excluded {num_excluded} files. {len(prs_files_to_merge)} files will be merged.")

    if not prs_files_to_merge:
        raise ValueError("After exclusion, no files are left to merge.")

    # --- C. Read all filtered files into a list of DataFrames ---
    dfs_to_merge = [pd.read_csv(f, sep='\t') for f in prs_files_to_merge]

    # --- D. Iteratively merge all DataFrames using an outer join ---
    # The reduce function applies the pd.merge command sequentially to the list of dataframes
    logger.info("Merging all remaining PRS files on 'eid'...")
    merged_prs_df = reduce(lambda left, right: pd.merge(left, right, on='eid', how='outer'), dfs_to_merge)
    logger.info(f"Merge complete. Final table has {merged_prs_df.shape[0]} rows and {merged_prs_df.shape[1]} columns.")

    # --- E. Save the final merged table ---
    output_filename = os.path.join(output_dir, 'PolygenicScores.csv')
    merged_prs_df.to_csv(output_filename, index=False)
    logger.info(f"Final combined PRS table saved to: {output_filename}")

    # --- F. Calculate and log missing percentage for each column ---
    logger.info("Calculating missing values report...")
    missing_df = merged_prs_df.isnull().mean().reset_index()
    missing_df.columns = ['variable', 'missing_percentage']
    missing_df['missing_percentage'] *= 100

    # Record end time and total duration
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"--- Process finished successfully ---")
    logger.info(f"Total time: {total_time:.3f} seconds")
    logger.info("\n------ Missing Values Report ------\n")
    logger.info(missing_df.to_string(index=False))

except Exception as e:
    logger.error(f"An error occurred during the process: {e}")
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Process terminated after {total_time:.3f} seconds due to error.")