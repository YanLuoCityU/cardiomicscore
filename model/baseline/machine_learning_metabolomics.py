import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm

# Import models
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import metrics and utilities
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score
from lifelines.utils import concordance_index
from sklearn.model_selection import ParameterSampler

# --- Project Setup ---
project_path = '/your path/cardiomicscore'
if project_path not in sys.path:
    sys.path.append(project_path)
    
# ==========================================================================================
# --- Helper Functions ---
# ==========================================================================================

def setup_logger(log_filename):
    """Sets up a file and stream logger."""
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger("ML_Training_Pipeline")

def hyperparameter_tuning_xgb(X_train, e_train, X_val, e_val, scale_pos_weight, n_iter=50, random_state=42):
    best_score = -np.inf
    best_params = None
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8], "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5, 1.0], "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [1000]
    }
    param_samples = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    for params in tqdm(param_samples, desc="Tuning XGBoost"):
        params["scale_pos_weight"] = scale_pos_weight
        model = xgb.XGBClassifier(**params, use_label_encoder=False, objective='binary:logistic', eval_metric='auc', 
                                  tree_method='hist', early_stopping_rounds=20, random_state=random_state)
        model.fit(X_train, e_train, eval_set=[(X_val, e_val)], verbose=False)
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(e_val, pred)
        if auc > best_score:
            best_score = auc
            best_params = params
    return best_params, best_score

def hyperparameter_tuning_lgbm(X_train, e_train, X_val, e_val, scale_pos_weight, n_iter=50, random_state=42):
    best_score = -np.inf
    best_params = None
    param_grid = {
        "num_leaves": [15, 31, 63], "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1], "n_estimators": [1000],
        "min_child_samples": [20, 30, 50], "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    param_samples = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    for params in tqdm(param_samples, desc="Tuning LightGBM"):
        params["scale_pos_weight"] = scale_pos_weight
        model = lgb.LGBMClassifier(**params, random_state=random_state, verbose=-1)
        model.fit(X_train, e_train, eval_set=[(X_val, e_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(e_val, pred)
        if auc > best_score:
            best_score = auc
            best_params = params
    return best_params, best_score

def hyperparameter_tuning_rf(X_train, e_train, X_val, e_val, n_iter=50, random_state=42):
    """
    Performs hyperparameter tuning for a RandomForestClassifier.
    Note: RandomForestClassifier uses 'class_weight' instead of 'scale_pos_weight'.
    """
    best_score = -np.inf
    best_params = None
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['sqrt', 'log2']
    }
    param_samples = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    for params in tqdm(param_samples, desc="Tuning Random Forest"):
        model = RandomForestClassifier(**params, class_weight='balanced', random_state=random_state, n_jobs=-1)
        model.fit(X_train, e_train)
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(e_val, pred)
        if auc > best_score:
            best_score = auc
            best_params = params
    return best_params, best_score

def hyperparameter_tuning_lr(X_train, e_train, X_val, e_val, random_state=42):
    best_score = -np.inf
    best_params = None
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    for C in tqdm(param_grid['C'], desc="Tuning Logistic Regression"):
        model = LogisticRegression(solver='liblinear', C=C, class_weight='balanced', random_state=random_state, max_iter=1000)
        model.fit(X_train, e_train)
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(e_val, pred)
        if auc > best_score:
            best_score = auc
            best_params = {'C': C}
    return best_params, best_score

def evaluate_model(model, X_test, y_test, e_test, outcome_name, logger):
    if len(X_test) == 0:
        logger.warning(f"Test set for {outcome_name} is empty. Skipping evaluation.")
        return None
    try:
        y_scores = model.predict_proba(X_test)[:, 1]
        y_binary = (y_scores >= 0.5).astype(int)
        
        logger.info(f"Performance for {outcome_name}:")
        header = f"{'Metric':<12} | {'Value':<10}"
        logger.info(header); logger.info("-" * len(header))
        
        metrics = {
            'AUC': roc_auc_score(e_test, y_scores),
            'PRAUC': average_precision_score(e_test, y_scores),
            'C-Index': 1 - concordance_index(y_test, y_scores, e_test),
            'Precision': precision_score(e_test, y_binary, zero_division=0),
            'Recall': recall_score(e_test, y_binary, zero_division=0)
        }
        for name, value in metrics.items():
            logger.info(f"{name:<12} | {value:.4f}")

        tn, fp, fn, tp = confusion_matrix(e_test, y_binary).ravel()
        cm_str = (f"  - Confusion Matrix:\n           Predicted 0 | Predicted 1\n         ---------------------------\n  True 0 |   {tn:^7d}   |   {fp:^7d}   \n  True 1 |   {fn:^7d}   |   {tp:^7d}   \n")
        logger.info(cm_str)

    except Exception as e:
        logger.error(f"Could not calculate metrics for {outcome_name}: {e}")

# ==========================================================================================
# --- Main Execution Block ---
# ==========================================================================================
if __name__ == '__main__':
    # 1. Configuration (defined directly in the script)
    config = {
        "name": "MachineLearning",
        "data_dir": "/your path/cardiomicscore/data/",
        "model_dir": "/your path/cardiomicscore/saved/models/",
        "log_dir": "/your path/cardiomicscore/saved/log/",
        "results_dir": "/your path/cardiomicscore/saved/results/",
        "predictor_set": "Metabolomics",
        "outcomes_list": ["cad", "stroke", "hf", "af", "pad",  "vte"],
        "seed_to_split": 250901,
        "seed": 1234
    }

    # 2. Setup Environment
    SEED = config['seed']
    np.random.seed(SEED)
    
    log_dir = os.path.join(config['log_dir'], config['name'], config['predictor_set'], str(SEED))
    log_filename = os.path.join(log_dir, "training_log.log")
    logger = setup_logger(log_filename)

    start_time = time.time()
    
    # 3. Load Data Directly from Feather Files
    logger.info("--- Loading Data Directly from Feather Files ---")
    predictor_set, seed_to_split = config['predictor_set'], config['seed_to_split']
    split_data_path = os.path.join(config['data_dir'], f"split_seed-{seed_to_split}")

    X_train_df = pd.read_feather(os.path.join(split_data_path, f"X_train_{predictor_set}.feather")).set_index('eid')
    X_val_df = pd.read_feather(os.path.join(split_data_path, f"X_val_{predictor_set}.feather")).set_index('eid')
    X_internal_df = pd.read_feather(os.path.join(split_data_path, f"X_internal_test_{predictor_set}.feather")).set_index('eid')
    X_external_df = pd.read_feather(os.path.join(split_data_path, f"X_external_test_{predictor_set}.feather")).set_index('eid')

    y_train_df, e_train_df = pd.read_feather(os.path.join(split_data_path, f"y_train_{predictor_set}.feather")).set_index('eid'), pd.read_feather(os.path.join(split_data_path, f"e_train_{predictor_set}.feather")).set_index('eid')
    y_val_df, e_val_df = pd.read_feather(os.path.join(split_data_path, f"y_val_{predictor_set}.feather")).set_index('eid'), pd.read_feather(os.path.join(split_data_path, f"e_val_{predictor_set}.feather")).set_index('eid')
    y_internal_df, e_internal_df = pd.read_feather(os.path.join(split_data_path, f"y_internal_test_{predictor_set}.feather")).set_index('eid'), pd.read_feather(os.path.join(split_data_path, f"e_internal_test_{predictor_set}.feather")).set_index('eid')
    y_external_df, e_external_df = pd.read_feather(os.path.join(split_data_path, "y_external_test.feather")).set_index('eid'), pd.read_feather(os.path.join(split_data_path, "e_external_test.feather")).set_index('eid')
    
    logger.info("\n--- Aligning DataFrames by Common eids ---")
    
    # Align training set
    common_train_ids = X_train_df.index.intersection(y_train_df.index).intersection(e_train_df.index)
    logger.info(f"  - Training set: Found {len(common_train_ids)} common eids. Aligning data.")
    X_train_df = X_train_df.reindex(common_train_ids)
    y_train_df = y_train_df.reindex(common_train_ids)
    e_train_df = e_train_df.reindex(common_train_ids)

    # Align validation set
    common_val_ids = X_val_df.index.intersection(y_val_df.index).intersection(e_val_df.index)
    logger.info(f"  - Validation set: Found {len(common_val_ids)} common eids. Aligning data.")
    X_val_df = X_val_df.reindex(common_val_ids)
    y_val_df = y_val_df.reindex(common_val_ids)
    e_val_df = e_val_df.reindex(common_val_ids)

    # Align internal test set
    common_internal_ids = X_internal_df.index.intersection(y_internal_df.index).intersection(e_internal_df.index)
    logger.info(f"  - Internal test set: Found {len(common_internal_ids)} common eids. Aligning data.")
    X_internal_df = X_internal_df.reindex(common_internal_ids)
    y_internal_df = y_internal_df.reindex(common_internal_ids)
    e_internal_df = e_internal_df.reindex(common_internal_ids)

    # Align external test set
    common_external_ids = X_external_df.index.intersection(y_external_df.index).intersection(e_external_df.index)
    logger.info(f"  - External test set: Found {len(common_external_ids)} common eids. Aligning data.")
    X_external_df = X_external_df.reindex(common_external_ids)
    y_external_df = y_external_df.reindex(common_external_ids)
    e_external_df = e_external_df.reindex(common_external_ids)

    all_data = {
        'train': {'X': X_train_df.values, 'y': y_train_df.values, 'e': e_train_df.values, 'eids': X_train_df.index.values},
        'val': {'X': X_val_df.values, 'y': y_val_df.values, 'e': e_val_df.values, 'eids': X_val_df.index.values},
        'internal_test': {'X': X_internal_df.values, 'y': y_internal_df.values, 'e': e_internal_df.values, 'eids': X_internal_df.index.values},
        'external_test': {'X': X_external_df.values, 'y': y_external_df.values, 'e': e_external_df.values, 'eids': X_external_df.index.values}
    }

    # 4. Define Models
    models_and_tuners = {
        'XGBoost': {'tuner': hyperparameter_tuning_xgb, 'model_class': xgb.XGBClassifier},
        'LGBM': {'tuner': hyperparameter_tuning_lgbm, 'model_class': lgb.LGBMClassifier},
        'RandomForest': {'tuner': hyperparameter_tuning_rf, 'model_class': RandomForestClassifier},
        'LogisticRegression': {'tuner': hyperparameter_tuning_lr, 'model_class': LogisticRegression}
    }
    
    outcomes_list = config['outcomes_list']
    outcome_indices = {outcome: i for i, outcome in enumerate(outcomes_list)}
    
    # 5. Main Loop
    for model_name, details in models_and_tuners.items():
        logger.info("\n" + "="*80 + f"\n--- Processing Model: {model_name} ---\n" + "="*80)
        
        scores_dfs = {
            split_name: pd.DataFrame(data['eids'], columns=['eid'])
            for split_name, data in all_data.items()
        }

        for outcome in outcomes_list:
            logger.info(f"\n--- Starting workflow for outcome: {outcome.upper()} ---")
            idx = outcome_indices[outcome]
            
            X_train, e_train_outcome = all_data['train']['X'], all_data['train']['e'][:, idx]
            X_val, e_val_outcome = all_data['val']['X'], all_data['val']['e'][:, idx]

            neg_count, pos_count = np.sum(e_train_outcome == 0), np.sum(e_train_outcome == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            
            logger.info(f"Step 1: Hyperparameter tuning for {outcome}...")
            if model_name in ['LogisticRegression', 'RandomForest']:
                best_params, best_score = details['tuner'](X_train, e_train_outcome, X_val, e_val_outcome, random_state=SEED)
            else:
                best_params, best_score = details['tuner'](X_train, e_train_outcome, X_val, e_val_outcome, scale_pos_weight, random_state=SEED)
            logger.info(f"Best params for {outcome}: {best_params} (Validation AUC: {best_score:.4f})")

            logger.info(f"Step 2: Fitting final model for {outcome} on training data...")
            final_model = details['model_class'](**best_params, random_state=SEED)
            if model_name in ['XGBoost', 'LGBM']:
                final_model.set_params(scale_pos_weight=scale_pos_weight)
                if model_name == 'XGBoost': final_model.set_params(use_label_encoder=False, eval_metric='logloss')
                if model_name == 'LGBM': final_model.set_params(verbose=-1)
            elif model_name == 'LogisticRegression':
                final_model.set_params(solver='liblinear', max_iter=1000, class_weight='balanced')
            elif model_name == 'RandomForest':
                final_model.set_params(class_weight='balanced', n_jobs=-1)
            final_model.fit(X_train, e_train_outcome)

            logger.info(f"Step 3: Evaluating and scoring final model for {outcome}...")
            
            for split_name, data in all_data.items():
                if len(data['X']) > 0:
                    scores = final_model.predict_proba(data['X'])[:, 1]
                    scores_dfs[split_name][outcome] = scores

            # Evaluation calls on test sets
            logger.info("\n-- Internal Test Set Evaluation --")
            evaluate_model(final_model, all_data['internal_test']['X'], all_data['internal_test']['y'][:, idx], all_data['internal_test']['e'][:, idx], outcome, logger)
            logger.info("\n-- External Test Set Evaluation --")
            evaluate_model(final_model, all_data['external_test']['X'], all_data['external_test']['y'][:, idx], all_data['external_test']['e'][:, idx], outcome, logger)

        logger.info(f"\n--- Saving all scores for model: {model_name} ---")
        results_dir = config['results_dir']
        scores_path = os.path.join(results_dir, "Scores", model_name)
        os.makedirs(scores_path, exist_ok=True)
        
        for split_name, df in scores_dfs.items():
            scores_filename = os.path.join(scores_path, f"{split_name}_scores_{config['predictor_set']}.csv")
            df.to_csv(scores_filename, index=False)
            logger.info(f"Saved {split_name} scores to {scores_filename}")
        
    end_time = time.time()
    logger.info(f"\nTotal script execution time: {(end_time - start_time)/60:.2f} minutes")
    logger.info("--- SCRIPT FINISHED ---")