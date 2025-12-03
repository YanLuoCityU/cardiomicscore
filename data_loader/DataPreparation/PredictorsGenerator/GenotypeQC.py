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

# Set up logger
log_filename = os.path.join(log_dir, 'GenotypeDataQC.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
genotype_qc_df = pd.read_csv(data_dir + 'genotype_qc.csv', low_memory=False)

'''
Ref: 
- https://dnanexus.gitbook.io/uk-biobank-rap/science-corner/gwas-using-alzheimers-disease
- https://www.nature.com/articles/s41562-024-02078-1#Sec13
'''
target_FieldID = [
    '22001-0.0', # Genetic sex
    '22006-0.0', # Genetic ethnic grouping
    '22019-0.0', # Sex chromosome aneuploidy
    '22021-0.0', # Genetic kinship to other participants
    '22027-0.0' # Outliers for heterozygosity or missing rate
]
cols_22009 = [col for col in genotype_qc_df.columns if col.startswith('22009-0.') and 0 <= int(col.split('.')[-1]) <= 20] # Genetic principal components
target_FieldID = ['eid'] + target_FieldID
feature_df = genotype_qc_df[target_FieldID]

# Rename columns
rename_dict = {
    '22001-0.0': 'genetic_sex',
    '22006-0.0': 'genetic_ethnic_grouping',
    '22019-0.0': 'sex_chromosome_aneuploidy',
    '22021-0.0': 'genetic_kinship',
    '22027-0.0': 'outliers_heterozygosity_missing_rate'
}
rename_dict.update({
    f'22009-0.{i}': f'gpc_{i}' for i in range(1, 21)
})
feature_df.rename(columns=rename_dict, inplace=True)
feature_df.to_csv(output_dir + 'GenotypeDataQC.csv', index=False)

# Caclulate missing percentage
missing_df = feature_df.isnull().mean().reset_index()
missing_df.columns = ['variable', 'missing_percentage']
missing_df['missing_percentage'] *= 100

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')
logger.info("\nMissing Values Report:\n")
logger.info(missing_df.to_string(index=False))