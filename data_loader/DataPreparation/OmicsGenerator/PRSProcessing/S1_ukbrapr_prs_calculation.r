# ==============================================================================
#                 Batch Polygenic Risk Score (PRS) Calculation
# ==============================================================================
#
# Environment: UKB RAP Posit Workbench (RStudio)
#
# Description:
#   This script automatically reads multiple formatted GWAS summary statistics files,
#   calculates a PRS for each file using the ukbrapR::create_pgs() function,
#   and finally merges all PRS scores for all individuals into a single table.
#
# ==============================================================================

# --- 1. Setup ---
# Note: You only need to run these installation lines once per session/project.
# You can comment them out with a '#' after the first successful run.
remotes::install_github("lcpilling/ukbrapR")
install.packages("tidyverse")

# Load necessary libraries
library(ukbrapR)
library(tidyverse)

base_path <- "/home/rstudio-server/"

# Output directory for all PRS results (the script will create this automatically)
output_dir <- file.path(base_path, "prs_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Define the list of all PGS file IDs to be processed
# These are all the files you have in your home directory
pgs_map <- c(
  "cad"    = "PGS003438",
  "stroke" = "PGS005230",
  "af"     = "PGS004905",
  "va"     = "PMID39657596",
  "hf"     = "PGS003969",
  "pad"    = "PGS005158",
  "aaa"    = "PGS000753",
  "vte"    = "PGS000043"
)

# --- 2. Loop through each file and calculate the PRS ---

# Create an empty list to store the results of each PRS calculation
all_prs_results <- list()

print("--- Starting batch PRS calculation process ---")

for (disease_abbr in names(pgs_map)) {
  
  # Get the corresponding PGS file ID from the map
  pgs_id <- pgs_map[[disease_abbr]]
  
  # Construct the input and output filenames
  input_file <- file.path(base_path, paste0(pgs_id, "_formatted.txt"))
  output_prefix <- file.path(output_dir, disease_abbr) # e.g., prs_results/cad
  prs_name <- paste0(disease_abbr, "_prs")             # e.g., cad_prs
  
  # Use tryCatch to capture and report errors without stopping the entire script
  tryCatch({
    
    cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
    cat(paste("⏳ Starting processing for:", disease_abbr, "(File:", pgs_id, ")\n"))
    
    # Check if the input file exists
    if (!file.exists(input_file)) {
      stop(paste("Input file not found:", input_file))
    }
    
    # Read the formatted SNP list
    varlist_pgs <- readr::read_tsv(input_file, show_col_types = FALSE)
    cat(paste("  - Read", nrow(varlist_pgs), "variants.\n"))
    
    # Call the core function from ukbrapR to calculate the PRS
    pgs_list <- create_pgs(
      in_file = varlist_pgs, 
      out_file = output_prefix,
      pgs_name = prs_name,
      progress = TRUE,
      very_verbose = TRUE,
      overwrite = TRUE
    )
      
    cat(paste("✅ Successfully processed:", pgs_id, "\n"))
    
  }, error = function(e) {
    # If an error occurs while processing a file, print the error message and continue to the next file
    cat(paste0("❌ ERROR: Failed to process ", pgs_id, ".\n"))
    cat("  - Error message:", conditionMessage(e), "\n")
  })
}