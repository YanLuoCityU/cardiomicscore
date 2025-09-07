# --- 1. Load Libraries ---
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(dplyr)
  library(cmprsk)
  library(survival)
  library(futile.logger)
  library(foreach)
  library(doParallel)
})

# --- 2. Function to Load and Prepare Data ---
load_and_prepare_data <- function(predictor_list, outcome, data_split, paths) {
  flog.info("Loading '%s' data for combination: [%s] and outcome: %s",
            data_split, paste(predictor_list, collapse=", "), outcome)
  death_path <- file.path(paths$DEATH_DATA_DIR, "OutcomesBasicInfo.csv")
  if (!file.exists(death_path)) { flog.error("Death data file not found: %s", death_path); return(data.table()) }
  death_df <- fread(death_path, select = c("eid", "bl2death_yrs"))
  y_path <- file.path(paths$SPLIT_SEED_PATH, sprintf("y_%s.feather", data_split))
  e_path <- file.path(paths$SPLIT_SEED_PATH, sprintf("e_%s.feather", data_split))
  if (!file.exists(y_path) || !file.exists(e_path)) { flog.error("Outcome (y/e) files not found for data split '%s'", data_split); return(data.table()) }
  y_df <- as.data.table(read_feather(y_path)); e_df <- as.data.table(read_feather(e_path))
  duration_col <- paste0('bl2', outcome, '_yrs'); event_col <- outcome
  y_df_renamed <- y_df[, .(eid, duration = get(duration_col))]; e_df_renamed <- e_df[, .(eid, event = get(event_col))]
  merged_df <- merge(y_df_renamed, e_df_renamed, by = 'eid'); merged_df <- merge(merged_df, death_df, by = 'eid', all.x = TRUE)
  for (predictor in predictor_list) {
    x_df <- data.table(); path <- "N/A"
    tryCatch({
      if (predictor == 'PANEL') {
        path <- file.path(paths$SPLIT_SEED_PATH, sprintf("X_%s_PANEL.feather", data_split)); x_df <- as.data.table(read_feather(path))
      } else if (predictor == 'Genomics') {
        path <- file.path(paths$SPLIT_SEED_PATH, sprintf("X_%s_Genomics.feather", data_split)); genomics_df <- as.data.table(read_feather(path))
        prs_col <- paste0(outcome, '_prs'); x_df <- genomics_df[, .(eid, prs = get(prs_col))]
      } else if (predictor %in% c('Metabolomics', 'Proteomics')) {
        path <- file.path(paths$SCORES_DATA_PATH, sprintf("%s_scores_%s.csv", data_split, predictor)); scores_df <- fread(path)
        final_name <- if (predictor == 'Metabolomics') 'metscore' else 'proscore'; x_df <- scores_df[, .(eid, temp_score = get(outcome))]
        x_df[, (final_name) := as.numeric(scale(temp_score))]; x_df[, temp_score := NULL]
      } else { flog.warn("Predictor '%s' has no defined loading logic. Skipping.", predictor); next }
      if (nrow(x_df) > 0) { merged_df <- merge(merged_df, x_df, by = 'eid', all = FALSE) } else { flog.error("Data for predictor '%s' was empty.", predictor); return(data.table()) }
    }, error = function(e) { flog.error("Failed to load data for predictor '%s'. Path: %s. Error: %s", predictor, path, e$message); return(data.table()) })
  }
  flog.info("Loaded %d common samples for [%s].", nrow(merged_df), paste(predictor_list, collapse=", ")); return(merged_df)
}

# --- 3. Function for a Single Bootstrap Iteration  ---
run_bootstrap_iteration <- function(iter_num, df_base_cr, df_combo_cr) {
  set.seed(iter_num)
  sample_indices <- sample(1:nrow(df_base_cr), size = nrow(df_base_cr), replace = TRUE)
  df_base_boot <- df_base_cr[sample_indices, ]; df_combo_boot <- df_combo_cr[sample_indices, ]
  if (nrow(df_combo_boot) < 50 || sum(df_combo_boot$fstatus == 1) < 5) { return(NULL) }
  tryCatch({
    features_to_exclude <- c('eid', 'duration', 'event', 'bl2death_yrs', 'death_time', 'ftime', 'fstatus')
    base_features <- setdiff(names(df_base_boot), features_to_exclude); combo_features <- setdiff(names(df_combo_boot), features_to_exclude)
    cov_base <- model.matrix(~ . - 1, data = df_base_boot[, ..base_features]); cov_combo <- model.matrix(~ . - 1, data = df_combo_boot[, ..combo_features])
    crr_base <- cmprsk::crr(df_base_boot$ftime, df_base_boot$fstatus, cov_base, failcode = 1, cencode = 0)
    crr_combo <- cmprsk::crr(df_combo_boot$ftime, df_combo_boot$fstatus, cov_combo, failcode = 1, cencode = 0)
    risk_score_base <- cov_base %*% crr_base$coef; risk_score_combo <- cov_combo %*% crr_combo$coef
    surv_obj_base <- survival::Surv(df_base_boot$ftime, df_base_boot$fstatus == 1); surv_obj_combo <- survival::Surv(df_combo_boot$ftime, df_combo_boot$fstatus == 1)
    c_index_base <- 1 - survival::concordance(surv_obj_base ~ risk_score_base)$concordance
    c_index_combo <- 1 - survival::concordance(surv_obj_combo ~ risk_score_combo)$concordance
    return(data.table(c_index_base=c_index_base, c_index_combo=c_index_combo, delta_c_index = c_index_combo - c_index_base))
  }, error = function(e) { return(NULL) })
}

# --- 4. Main Analysis Function ---
run_all_comparisons_bootstrap <- function(paths, outcomes_to_run, file_suffix) {
  flog.info('--- Starting Bootstrap Confidence Interval Estimation for Competing Risks ---')

  DATA_SPLIT <- 'external_test'
  predictor_sets <- list(
    c('PANEL'), c('PANEL', 'Genomics'), c('PANEL', 'Metabolomics'), c('PANEL', 'Proteomics'),
    c('PANEL', 'Genomics', 'Metabolomics'), c('PANEL', 'Genomics', 'Proteomics'),
    c('PANEL', 'Metabolomics', 'Proteomics'), c('PANEL', 'Genomics', 'Metabolomics', 'Proteomics'),
    c('Genomics'), c('Metabolomics'), c('Proteomics')
  )
  total_predictor_sets <- length(predictor_sets)
  all_results_list <- list()
  n_bootstraps <- 1000; n_cores <- 80
  cl <- parallel::makeCluster(n_cores); doParallel::registerDoParallel(cl)
  flog.info("Setup parallel processing with %d cores for %d bootstraps.", n_cores, n_bootstraps)

  for (outcome in outcomes_to_run) {
    outcome_start_time <- Sys.time()
    flog.info("\n%s PROCESSING OUTCOME: %s %s", paste(rep('#', 20), collapse=''), toupper(outcome), paste(rep('#', 20), collapse=''))

    baseline_combo <- list('PANEL')
    df_base_full <- load_and_prepare_data(baseline_combo[[1]], outcome, DATA_SPLIT, paths)
    if (nrow(df_base_full) == 0) { flog.error("Could not load 'PANEL' data for %s. Skipping.", outcome); next }

    for (i in 1:total_predictor_sets) {
      combo <- predictor_sets[[i]]; combo_name <- paste(combo, collapse = "_"); baseline_name <- paste(baseline_combo[[1]], collapse="_")
      flog.info("\n----- [Outcome: %s | Set: %d/%d] -----", toupper(outcome), i, total_predictor_sets)
      flog.info("Bootstrapping Combo: [%s] vs. Baseline: [%s]", combo_name, baseline_name)
      
      df_combo_full <- load_and_prepare_data(combo, outcome, DATA_SPLIT, paths)
      if (nrow(df_combo_full) == 0) { flog.warn("Data for combo %s is empty. Skipping.", combo_name); next }

      common_eids <- intersect(df_base_full$eid, df_combo_full$eid)
      df_base_common <- df_base_full[eid %in% common_eids]; df_combo_common <- df_combo_full[eid %in% common_eids]
      n_samples <- length(common_eids)
      flog.info("Using %d common samples for comparison.", n_samples)
      
      prepare_cr_data <- function(df) {
        dt <- copy(df); dt[, death_time := fifelse(is.na(bl2death_yrs), Inf, bl2death_yrs)]; dt[, ftime := pmin(duration, death_time, na.rm = TRUE)]
        dt[, fstatus := case_when(event == 1 & ftime == duration ~ 1, !is.infinite(death_time) & ftime == death_time ~ 2, TRUE ~ 0)]; return(dt)
      }
      df_base_cr <- prepare_cr_data(df_base_common); df_combo_cr <- prepare_cr_data(df_combo_common)

      bootstrap_results_dt <- foreach(iter = 1:n_bootstraps, .combine = 'rbind', .packages = c('data.table', 'cmprsk', 'survival'), .export = c('run_bootstrap_iteration')) %dopar% {
        run_bootstrap_iteration(iter, df_base_cr, df_combo_cr)
      }

      valid_results <- na.omit(bootstrap_results_dt)
      flog.info("Completed %d valid bootstrap iterations.", nrow(valid_results))

      if (nrow(valid_results) > 20) {
        ci_summary <- valid_results[, .(c_index_combo_mean = mean(c_index_combo), c_index_combo_ci_lower = quantile(c_index_combo, 0.025), c_index_combo_ci_upper = quantile(c_index_combo, 0.975),
                                        delta_c_index_mean = mean(delta_c_index), delta_c_index_ci_lower = quantile(delta_c_index, 0.025), delta_c_index_ci_upper = quantile(delta_c_index, 0.975))]
        final_result <- list(outcome = outcome, baseline_model = baseline_name, comparison_model = combo_name, n_samples = n_samples)
        all_results_list[[length(all_results_list) + 1]] <- c(final_result, as.list(ci_summary))
      } else { flog.warn("Not enough valid bootstrap iterations for combo %s.", combo_name) }
    }
    
    outcome_end_time <- Sys.time()
    outcome_elapsed_time <- difftime(outcome_end_time, outcome_start_time, units = "mins")
    flog.info("\n%s COMPLETED OUTCOME: %s %s", paste(rep('#', 20), collapse=''), toupper(outcome), paste(rep('#', 20), collapse=''))
    flog.info("Time taken for this outcome: %.2f minutes.", as.numeric(outcome_elapsed_time))
  }

  parallel::stopCluster(cl)

  if (length(all_results_list) > 0) {
    results_df <- rbindlist(all_results_list)
    col_order <- c('outcome', 'baseline_model', 'comparison_model', 'n_samples', 'c_index_combo_mean', 'c_index_combo_ci_lower', 'c_index_combo_ci_upper', 
                   'delta_c_index_mean', 'delta_c_index_ci_lower', 'delta_c_index_ci_upper')
    results_df <- results_df[, ..col_order]
    
    results_filename <- file.path(paths$CINDEX_SAVE_DIR, sprintf("cindex_competing_risk_bootstrap_ci_%s.csv", file_suffix))
    fwrite(results_df, results_filename, row.names = FALSE, quote = FALSE)
    flog.info("\n\nSUCCESS: All analyses complete. Results saved to %s", results_filename)
  } else {
    flog.warn("No bootstrap results were generated.")
  }
}

# --- 5. Script Entry Point ---
main <- function() {
  script_start_time <- Sys.time()

  OUTCOMES_TO_RUN <- c("cad", "stroke")
  FILE_SUFFIX <- paste(OUTCOMES_TO_RUN, collapse="_")

  # Configuration
  BASE_DIR <- "/your path/cardiomicscore"
  SEED_TO_SPLIT <- 250901
  paths <- list(DATA_DIR = file.path(BASE_DIR, "data"), RESULTS_DIR = file.path(BASE_DIR, "saved/results"), LOG_DIR = file.path(BASE_DIR, "saved/log"),
                DEATH_DATA_DIR = file.path(BASE_DIR, "data/processed/outcomes"), SPLIT_SEED_PATH = file.path(BASE_DIR, "data", paste0("split_seed-", SEED_TO_SPLIT)))
  paths$SCORES_DATA_PATH <- file.path(paths$RESULTS_DIR, 'Scores/OmicsNet/Final'); paths$CINDEX_SAVE_DIR <- file.path(paths$RESULTS_DIR, 'Cindex')

  dir.create(file.path(paths$LOG_DIR, "FineGray"), showWarnings = FALSE, recursive = TRUE)
  dir.create(paths$CINDEX_SAVE_DIR, showWarnings = FALSE, recursive = TRUE)
  log_file <- file.path(paths$LOG_DIR, "FineGray", sprintf("Bootstrap_CI_CR_%s.log", FILE_SUFFIX))
  flog.appender(appender.tee(log_file)); flog.layout(layout.format('[~t] [~l] ~m'))
  
  run_all_comparisons_bootstrap(paths = paths, outcomes_to_run = OUTCOMES_TO_RUN, file_suffix = FILE_SUFFIX)
  
  total_elapsed_time <- difftime(Sys.time(), script_start_time, units = "mins")
  flog.info("\n--- Total Script Execution Time: %.2f minutes ---", as.numeric(total_elapsed_time))
}

# Run the main function
main()