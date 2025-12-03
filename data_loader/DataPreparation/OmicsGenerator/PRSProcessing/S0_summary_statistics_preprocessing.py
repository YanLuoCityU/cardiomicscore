# ==============================================================================
#           Unified and Optimized GWAS/PGS Summary Statistics Formatter
# ==============================================================================

import pandas as pd
import numpy as np
import requests
import sys
from tqdm import tqdm

# --- Reusable Helper Module 1: Ensembl API Fetcher ---
def get_ensembl_info(rsid):
    """
    Queries the Ensembl GRCh37 API and intelligently selects the mapping on a
    primary chromosome (1-22, X, Y) to get chr, pos, and alleles.
    """
    server = "https://grch37.rest.ensembl.org"
    ext = f"/variation/human/{rsid}?content-type=application/json"
    try:
        r = requests.get(server + ext, headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        primary_chromosomes = {str(i) for i in range(1, 23)} | {'X', 'Y'}

        for mapping in data.get('mappings', []):
            chrom = mapping.get('seq_region_name')
            if chrom in primary_chromosomes:
                pos = mapping.get('start')
                alleles = mapping.get('allele_string', '').split('/')
                if pos and alleles and alleles[0]:
                    return {'api_chr': chrom, 'api_pos': pos, 'api_alleles': alleles}
        
        return None
    except (requests.exceptions.RequestException, ValueError):
        return None

# --- Main Processing Function ---
def process_gwas_file(input_path, output_path, manual_overrides=None):
    """
    A single function to intelligently process any of the given GWAS/PGS files.
    """
    print(f"\n{'='*60}\nProcessing file: {input_path}\n{'='*60}")
    
    try:
        # --- 1. Read File with Auto-detected Separator ---
        print("Reading file...")
        with open(input_path, 'r') as f:
            header = ''
            for line in f:
                if not line.startswith('#'):
                    header = line
                    break
        separator = '\t' if '\t' in header else '\s+'
        separator_name = 'tab' if separator == '\t' else 'whitespace'
        print(f"Auto-detected separator: '{separator_name}'")
        
        df = pd.read_csv(input_path, comment='#', sep=separator, engine='python')
        print("File read successfully.")

        # --- 2. Auto-detect and Enrich Data if Necessary ---
        if 'chr_position' not in df.columns and 'Position' not in df.columns:
            print("Position data missing. Fetching from Ensembl API...")
            api_results = [get_ensembl_info(rsid) for rsid in tqdm(df['rsID'])]
            df_api = pd.DataFrame([res for res in api_results if res is not None])
            df = pd.concat([df.reset_index(drop=True), df_api], axis=1)
            
            if 'chr_name' in df.columns:
                df = df.drop(columns=['chr_name'])
            if 'Chromosome' in df.columns:
                df = df.drop(columns=['Chromosome'])

            def find_other_allele(row):
                if isinstance(row.get('api_alleles'), list) and len(row['api_alleles']) > 1:
                    found = [a for a in row['api_alleles'] if a != row['effect_allele']]
                    return found[0] if found else None
                return None
            df['other_allele'] = df.apply(find_other_allele, axis=1)
        
        if 'effect_weight' not in df.columns and 'Ors' in df.columns:
            print("Effect weight (beta) missing. Calculating from Odds Ratio (Ors)...")
            df['effect_weight'] = np.log(df['Ors'])

        # --- 3. Standardize Column Names ---
        rename_map = {
            'rsID': 'rsid', 'SNP': 'rsid',
            'chr_name': 'chr', 'Chromosome': 'chr', 'api_chr': 'chr',
            'chr_position': 'pos', 'Position': 'pos', 'api_pos': 'pos',
            'effect_allele': 'effect_allele', 'Effect allele': 'effect_allele',
            'other_allele': 'other_allele', 'Other allele': 'other_allele',
            'effect_weight': 'beta'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # --- 4. Apply Manual Overrides if provided ---
        if manual_overrides:
            print("Applying manual overrides...")
            for snp_id, actions in manual_overrides.items():
                if actions.get('action') == 'remove':
                    df = df[df['rsid'] != snp_id].copy()
                    print(f" - Removed SNP: {snp_id}")
                else:
                    for col, value in actions.items():
                        df.loc[df['rsid'] == snp_id, col] = value
                        print(f" - For {snp_id}, set '{col}' to '{value}'")

        # --- 5. Finalize and Save ---
        final_columns = ['rsid', 'chr', 'pos', 'effect_allele', 'other_allele', 'beta']
        missing_cols = [col for col in final_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Could not produce all required final columns. Missing: {missing_cols}")
            
        df_final = df[final_columns]

        print("Sorting the final data by chromosome and rsid...")
        # Create a temporary numeric column for robust chromosome sorting (handles 'X', 'Y')
        df_final['chr_sort_key'] = pd.to_numeric(df_final['chr'].replace({'X': 23, 'Y': 24, 'MT': 25}), errors='coerce')
        
        # Sort by the numeric chromosome key, then by rsid, and drop the temporary key
        df_final = df_final.sort_values(by=['chr_sort_key', 'rsid']).drop(columns=['chr_sort_key'])
        
        df_final.to_csv(output_path, sep='\t', index=False)
        print(f"\n🎉 Processing complete! Formatted file saved to: {output_path}")
        print("Final data preview:")
        print(df_final.head())
        
    except Exception as e:
        print(f"\n❌ An error occurred while processing {input_path}: {e}", file=sys.stderr)

# ==============================================================================
#                               Execution Block
# ==============================================================================
if __name__ == "__main__":
    base_path = "/home/luoyan/phd_project/MultiomicsCVDv2/data/gwas_summary_statistics/"
    
    # NOTE on SNP Removals:
    # All SNPs marked with {'action': 'remove'} were verified to be problematic
    # on the UKB RAP Posit Workbench (RStudio) platform using the ukbrapR package.
    # These variants are typically indels that fail during the PLINK merge step,
    # or they fail quality control (i.e., they are missing from the imputed BGEN files).
    #
    # Example verification code snippet in R:
    #   library(ukbrapR)
    #   varlist <- data.frame(rsid=c("rsXXX", "rsYYY"), chr=c(1, 2))
    #   imputed_genotypes <- extract_genotypes(varlist) # This step would fail or return empty for the problematic SNP.
    
    files_to_process = {
        'PGS003438': { # CAD
            'overrides': {
                'rs582384': {'action': 'remove'},      # Reason: multiallelic variant
            } 
        },
        'PGS005230': { # Stroke
            'overrides': {
                'rs7314740': {'action': 'remove'},     # Reason: multiallelic variant
                'rs79485249': {'action': 'remove'}     # Reason: multiallelic variant
            }
        },
        'PGS004905': { # AF
            'overrides': {'rs6771054': {'action': 'remove'}} # Reason: multiallelic variant
        }, 
        'PMID39657596': { # VA
            'overrides': {'rs7124547': {'action': 'remove'}} # Reason: multiallelic variant
        }, 
        'PGS005158': { # PAD
             # Reason: Manual correction of 'other_allele' based on original publication
             # Ref: https://www.nature.com/articles/s41591-019-0492-5/tables/1
            'overrides': {'rs62084752': {'other_allele': 'G'}}
        },
        'PGS000753': {}, # AAA
        'PGS000043': {}  # VTE
    }
    
    for name, config in files_to_process.items():
        input_file = f"{base_path}{name}.txt"
        output_file = f"{base_path}{name}_formatted.txt"
        
        process_gwas_file(
            input_path=input_file,
            output_path=output_file,
            manual_overrides=config.get('overrides')
        )