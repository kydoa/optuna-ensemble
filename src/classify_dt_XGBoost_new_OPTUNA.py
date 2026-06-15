import os
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import optuna
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURAÇÃO FUNDAMENTAL
# ==============================================================================
BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST     = 'ADReSSo21_test'

# --- FUNÇÃO DE TUNAGEM (OPTUNA) ---
def objective(trial, X_train, y_train):
    if len(X_train) <= 3:
        return 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train)

    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 6), # Capped at 6 for high-dim stability
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': 1
    }

    num_boost_round = trial.suggest_int('n_estimators', 50, 150) # Capped at 150 to fly through ComParE

    try:
        cv_results = xgb.cv(
            param,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=3,
            stratified=True,
            early_stopping_rounds=10,
            seed=42
        )
        best_score = 1.0 - cv_results['test-logloss-mean'].iloc[-1]
        return best_score
    except Exception:
        return 0.0

# --- FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO ---
def load_data_for_feature_set(feature_folder, dataset_name):
    target_path = BASE_FEATURES_DIR.joinpath(feature_folder, dataset_name)
    features, labels, ids = [], [], []
    if not target_path.exists():
        return np.array([]), [], []

    expected_dim = None
    for root, dirs, files in os.walk(target_path):
        if not files: continue
        folder_name = Path(root).name.upper()
        current_label_str = None

        if any(kw in folder_name for kw in ['HC', 'CONTROL', 'CN', 'HEALTHY', 'NORMAL']):
            current_label_str = 'HC'
        elif any(kw in folder_name for kw in ['AD', 'ALZ', 'PATIENT', 'DEMENTIA']):
            current_label_str = 'AD'
        else:
            for f in files[:1]:
                f_low = f.lower()
                if 'hc' in f_low or 'control' in f_low:
                    current_label_str = 'HC'
                elif 'ad' in f_low or 'alz' in f_low:
                    current_label_str = 'AD'

        if not current_label_str: continue
        numeric_label = 1 if current_label_str == 'AD' else 0

        for file in files:
            if file.endswith('.npy'):
                try:
                    data = np.load(os.path.join(root, file)).flatten().astype(np.float32)
                    if expected_dim is None:
                        expected_dim = data.size
                    elif data.size != expected_dim:
                        continue

                    if np.isnan(data).all(): continue

                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(numeric_label)
                except Exception:
                    pass

    if not features:
        return np.array([]), [], []
    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Worker function used by Parallel."""
    print(f"[>] Processing {feature_folder}")

    X_train, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)
    if len(X_train) <= 3 or len(np.unique(y_train)) < 2:
        print(f"    [!] Skipping {feature_folder}: Insufficient samples.")
        return None

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0 or X_test.shape[1] != X_train.shape[1]:
        return None

    # --- DEFENSOR DE PERFORMANCE: FILTRO PARA ALTA DIMENSIONALIDADE ---
    if X_train.shape[1] > 500:
        print(f"    [i] High-dim detected ({X_train.shape[1]} features). Applying fast filtering...")

        selector_var = VarianceThreshold(threshold=0.01)
        X_train = selector_var.fit_transform(X_train)
        X_test = selector_var.transform(X_test)

        k_features = min(200, X_train.shape[1])
        selector_k = SelectKBest(score_func=f_classif, k=k_features)
        X_train = selector_k.fit_transform(X_train, y_train)
        X_test = selector_k.transform(X_test)
        print(f"    [i] Successfully optimized matrix shape to: {X_train.shape}")

    # --- INÍCIO DA TUNAGEM COM OPTUNA ---
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=135)
    # --- FIM DA TUNAGEM ---

    clf = XGBClassifier(**study.best_params, eval_metric='logloss', random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    prob_dict, label_dict = {}, {}
    for i in range(len(test_ids)):
        p_id = test_ids[i]
        prob_dict[p_id] = probs[i, 1] if probs.shape[1] > 1 else probs[i, 0]
        label_dict[p_id] = int(y_test[i])

    print(f"    [OK] Feature {feature_folder} finalized!")
    return feature_folder, prob_dict, label_dict

def run_pipeline():
    print("="*70)
    print("INICIANDO PIPELINE DE ENSEMBLE (OPTIMIZED HIGH-DIM FOR COMPARE)")
    print("="*70)

    if not BASE_FEATURES_DIR.exists():
        return

    all_features = [f for f in os.listdir(BASE_FEATURES_DIR)
                    if os.path.isdir(BASE_FEATURES_DIR.joinpath(f))]

    print(f"[INFO] Distributing {len(all_features)} features to CPU cores...")

    # Alterado backend para 'loky' (multiprocessing padrão) para contornar o travamento do GIL em threads
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_single_feature_set)(feat, DATASET_TRAIN, DATA_TEST)
        for feat in all_features
    )

    # --- AGGREGATION PHASE ---
    patient_probs_accumulator = {}
    true_labels_map = {}
    features_processed_count = 0
    all_test_ids = set()

    print("[INFO] Aggregating results...")
    for res in results:
        if res is None: continue
        feat_name, p_probs, p_labels = res
        features_processed_count += 1
        for p_id, prob in p_probs.items():
            all_test_ids.add(p_id)
            patient_probs_accumulator.setdefault(p_id, []).append(prob)
        for p_id, label in p_labels.items():
            true_labels_map[p_id] = label

    # --- FINAL VOTING (Soft Voting) ---
    final_predictions, final_ground_truth, ids_intersecao = [], [], []
    for p_id in all_test_ids:
        probs_list = patient_probs_accumulator.get(p_id, [])
        if len(probs_list) == features_processed_count:
            avg_prob_ad = np.mean(probs_list)
            pred_bin = 1 if avg_prob_ad > 0.5 else 0
            true_val = true_labels_map.get(p_id, -1)
            if true_val != -1:
                final_predictions.append(pred_bin)
                final_ground_truth.append(true_val)
                ids_intersecao.append(p_id)

    # ==========================================================================
    # SUPORTE AO JUPYTER NOTEBOOK INSERIDO AQUI
    # ==========================================================================
    if final_predictions:
        df_results = pd.DataFrame({
            'Patient_ID': ids_intersecao,
            'True_Label': final_ground_truth,
            'Predicted_Class': final_predictions
        })
        df_results.to_csv('/home/dani/Documentos/PEP/ensemble_results_XG.csv', index=False)
        print("[INFO] Resultados exportados com sucesso em 'ensemble_results_XG.csv'!")
    # ==========================================================================

    if not final_predictions:
        print("[!] Erro: Nenhum paciente comum encontrado.")
        return

    print("\n" + "="*70)
    print(f"Acurácia Global do Ensemble: {accuracy_score(final_ground_truth, final_predictions):.4f}")
    print("-" * 70)
    print("Relatório de Classificação:")
    print(classification_report(final_ground_truth, final_predictions,
                                labels=[0, 1], target_names=['HC', 'AD'], zero_division=0))
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
