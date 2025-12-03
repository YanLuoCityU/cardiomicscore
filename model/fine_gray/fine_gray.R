# --- 1. Load Libraries ---
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(dplyr)
  library(cmprsk)
  library(survival)
  library(futile.logger)
})

# --- 2. Function to Load and Prepare Data ---
# Adapted from coxph.py to handle multiple outcomes and specific predictor types
load_and_prepare_data <- function(predictor_list, outcome, data_split, paths) {
  flog.info("Loading '%s' data for combination: [%s] and outcome: %s",
            data_split, paste(predictor_list, collapse=", "), outcome)

  # Load competing risk event data (death)
  death_path <- file.path(paths$DEATH_DATA_DIR, "OutcomesBasicInfo.csv")
  if (!file.exists(death_path)) {
    flog.error("Death data file not found: %s", death_path)
    return(data.table())
  }
  death_df <- fread(death_path, select = c("eid", "bl2death_yrs"))

  # Load primary event and duration data (y and e files)
  y_path <- file.path(paths$SPLIT_SEED_PATH, sprintf("y_%s.feather", data_split))
  e_path <- file.path(paths$SPLIT_SEED_PATH, sprintf("e_%s.feather", data_split))
  if (!file.exists(y_path) || !file.exists(e_path)) {
    flog.error("Outcome (y/e) files not found for data split '%s'", data_split)
    return(data.table())
  }
  y_df <- as.data.table(read_feather(y_path))
  e_df <- as.data.table(read_feather(e_path))

  duration_col <- paste0('bl2', outcome, '_yrs')
  event_col <- outcome
  y_df_renamed <- y_df[, .(eid, duration = get(duration_col))]
  e_df_renamed <- e_df[, .(eid, event = get(event_col))]
  merged_df <- merge(y_df_renamed, e_df_renamed, by = 'eid')
  merged_df <- merge(merged_df, death_df, by = 'eid', all.x = TRUE)

  # Load predictor data
  for (predictor in predictor_list) {
    x_df <- data.table()
    path <- "N/A"
    tryCatch({
      if (predictor == 'PANEL') {
        path <- file.path(paths$SPLIT_SEED_PATH, sprintf("X_%s_PANEL.feather", data_split))
        x_df <- as.data.table(read_feather(path))
      } else if (predictor == 'Genomics') {
        path <- file.path(paths$SPLIT_SEED_PATH, sprintf("X_%s_Genomics.feather", data_split))
        genomics_df <- as.data.table(read_feather(path))
        prs_col <- paste0(outcome, '_prs')
        x_df <- genomics_df[, .(eid, prs = get(prs_col))]
      } else if (predictor %in% c('Metabolomics', 'Proteomics')) {
        path <- file.path(paths$SCORES_DATA_PATH, sprintf("%s_scores_%s.csv", data_split, predictor))
        scores_df <- fread(path)
        final_name <- if (predictor == 'Metabolomics') 'metscore' else 'proscore'
        x_df <- scores_df[, .(eid, temp_score = get(outcome))]
        x_df[, (final_name) := as.numeric(scale(temp_score))]
        x_df[, temp_score := NULL]
      } else {
        flog.warn("Predictor '%s' has no defined loading logic. Skipping.", predictor)
        next
      }

      if (nrow(x_df) > 0) {
        merged_df <- merge(merged_df, x_df, by = 'eid', all = FALSE)
      } else {
        flog.error("Data for predictor '%s' was empty. Aborting load for this combo.", predictor)
        return(data.table())
      }
    }, error = function(e) {
      flog.error("Failed to load data for predictor '%s'. Path: %s. Error: %s", predictor, path, e$message)
      return(data.table())
    })
  }

  flog.info("Loaded %d common samples for [%s].", nrow(merged_df), paste(predictor_list, collapse=", "))
  return(merged_df)
}

# --- 3. Main Analysis Function ---
run_competing_risk_analysis <- function(paths) {
  flog.info("--- Starting Point Estimate Evaluation for Competing Risk Models ---")

  outcomes_to_run <- c('cad', 'stroke', 'hf', 'af', 'pad', 'vte')
  DATA_SPLIT <- 'external_test'
  
  # Predictor sets as specified
  predictor_sets <- list(
    c('PANEL'), c('PANEL', 'Genomics'), c('PANEL', 'Metabolomics'), c('PANEL', 'Proteomics'),
    c('PANEL', 'Genomics', 'Metabolomics'), c('PANEL', 'Genomics', 'Proteomics'),
    c('PANEL', 'Metabolomics', 'Proteomics'), c('PANEL', 'Genomics', 'Metabolomics', 'Proteomics'),
    c('Genomics'), c('Metabolomics'), c('Proteomics')
  )

  all_results_list <- list()

  for (outcome in outcomes_to_run) {
    flog.info("\n%s PROCESSING OUTCOME: %s %s", paste(rep('=', 20), collapse=''), toupper(outcome), paste(rep('=', 20), collapse=''))

    baseline_combo <- list('PANEL')
    df_base_full <- load_and_prepare_data(baseline_combo[[1]], outcome, DATA_SPLIT, paths)
    if (nrow(df_base_full) == 0) {
      flog.error("Could not load 'PANEL' baseline data for outcome %s. Skipping outcome.", outcome)
      next
    }

    for (combo in predictor_sets) {
      combo_name <- paste(combo, collapse = "_")
      baseline_name <- paste(baseline_combo[[1]], collapse="_")
      flog.info("\n----- Evaluating Combo: %s vs. Baseline: %s -----", combo_name, baseline_name)

      df_combo_full <- load_and_prepare_data(combo, outcome, DATA_SPLIT, paths)
      if (nrow(df_combo_full) == 0) {
        flog.warn("Data for combination %s is empty. Skipping.", combo_name)
        next
      }

      common_eids <- intersect(df_base_full$eid, df_combo_full$eid)
      df_base_common <- df_base_full[eid %in% common_eids]
      df_combo_common <- df_combo_full[eid %in% common_eids]
      n_samples <- length(common_eids)
      flog.info("Using %d common samples for comparison.", n_samples)

      deaths_in_common <- df_combo_common[!is.na(bl2death_yrs), .N]
      flog.info("Number of deaths within these common samples: %d", deaths_in_common)

      # Function to prepare data for competing risk analysis
      prepare_cr_data <- function(df) {
        dt <- copy(df)
        dt[, death_time := fifelse(is.na(bl2death_yrs), Inf, bl2death_yrs)]
        dt[, ftime := pmin(duration, death_time, na.rm = TRUE)]
        dt[, fstatus := case_when(
          event == 1 & ftime == duration ~ 1,      # Primary event
          !is.infinite(death_time) & ftime == death_time ~ 2, # Competing event
          TRUE ~ 0                                 # Censored
        )]
        return(dt)
      }
      df_base_cr <- prepare_cr_data(df_base_common)
      df_combo_cr <- prepare_cr_data(df_combo_common)

      primary_events_in_set <- df_base_cr[fstatus == 1, .N]
      competing_events_in_set <- df_base_cr[fstatus == 2, .N]
      flog.info("Primary events (fstatus=1) in analysis set: %d", primary_events_in_set)
      flog.info("Competing events (fstatus=2) in analysis set: %d", competing_events_in_set)

      tryCatch({
        if (sum(df_combo_cr$fstatus == 1) < 5) {
          flog.warn("Combination %s has fewer than 5 primary events. Skipping.", combo_name)
          next
        }

        # Define features and create model matrices
        features_to_exclude <- c('eid', 'duration', 'event', 'bl2death_yrs', 'death_time', 'ftime', 'fstatus')
        base_features <- setdiff(names(df_base_cr), features_to_exclude)
        combo_features <- setdiff(names(df_combo_cr), features_to_exclude)

        cov_base <- model.matrix(~ . - 1, data = df_base_cr[, ..base_features])
        cov_combo <- model.matrix(~ . - 1, data = df_combo_cr[, ..combo_features])

        # Fit Fine-Gray models
        crr_base <- crr(df_base_cr$ftime, df_base_cr$fstatus, cov_base, failcode = 1, cencode = 0)
        crr_combo <- crr(df_combo_cr$ftime, df_combo_cr$fstatus, cov_combo, failcode = 1, cencode = 0)

        # Calculate Harrell's C-index
        risk_score_base <- cov_base %*% crr_base$coef
        risk_score_combo <- cov_combo %*% crr_combo$coef
        surv_obj_base <- Surv(df_base_cr$ftime, df_base_cr$fstatus == 1)
        surv_obj_combo <- Surv(df_combo_cr$ftime, df_combo_cr$fstatus == 1)
        c_index_base <- 1 - concordance(surv_obj_base ~ risk_score_base)$concordance
        c_index_combo <- 1 - concordance(surv_obj_combo ~ risk_score_combo)$concordance
        delta_c_index <- c_index_combo - c_index_base

        # Store results in a list
        results <- list(
          outcome = outcome,
          baseline_model = baseline_name,
          comparison_model = combo_name,
          n_samples = n_samples,
          c_index_base = c_index_base,
          c_index_combo = c_index_combo,
          delta_c_index = delta_c_index
        )
        all_results_list[[length(all_results_list) + 1]] <- results
        flog.info("C-Index for %s: %.4f (Delta vs. %s: %+.4f)", combo_name, c_index_combo, baseline_name, delta_c_index)

      }, error = function(e) {
        flog.error("Calculation failed for combo %s, outcome %s: %s", combo_name, outcome, e$message)
      })
    }
  }

  # Consolidate and save all results to a single CSV file
  if (length(all_results_list) > 0) {
    results_df <- rbindlist(all_results_list)
    results_filename <- file.path(paths$CINDEX_SAVE_DIR, "cindex_competing_risk_summary.csv")
    fwrite(results_df, results_filename, row.names = FALSE, quote = FALSE)
    flog.info("\n\nSUCCESS: All analyses complete. Consolidated results saved to %s", results_filename)
  } else {
    flog.warn("No results were generated.")
  }
}

# --- 4. Script Entry Point ---
main <- function() {
  script_start_time <- Sys.time()

  # Configuration from coxph.py
  BASE_DIR <- "/your path/cardiomicscore"
  SEED_TO_SPLIT <- 250901

  paths <- list(
    DATA_DIR = file.path(BASE_DIR, "data"),
    RESULTS_DIR = file.path(BASE_DIR, "saved/results"),
    LOG_DIR = file.path(BASE_DIR, "saved/log"),
    DEATH_DATA_DIR = file.path(BASE_DIR, "data/processed/outcomes"),
    SPLIT_SEED_PATH = file.path(BASE_DIR, "data", paste0("split_seed-", SEED_TO_SPLIT))
  )
  paths$SCORES_DATA_PATH <- file.path(paths$RESULTS_DIR, 'Scores/OmicsNet/Final')
  paths$CINDEX_SAVE_DIR <- file.path(paths$RESULTS_DIR, 'Cindex')

  # Create directories
  dir.create(file.path(paths$LOG_DIR, "FineGray"), showWarnings = FALSE, recursive = TRUE)
  dir.create(paths$CINDEX_SAVE_DIR, showWarnings = FALSE, recursive = TRUE)

  # Setup logging
  log_file <- file.path(paths$LOG_DIR, "FineGray/Point_Estimates_CR.log")
  flog.appender(appender.tee(log_file))
  flog.layout(layout.format('[~t] [~l] ~m'))
  
  run_competing_risk_analysis(paths = paths)
  
  total_elapsed_time <- difftime(Sys.time(), script_start_time, units = "mins")
  flog.info("\n--- Total Script Execution Time: %.2f minutes ---", as.numeric(total_elapsed_time))
}

# Run the main function
main()