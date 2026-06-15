import os
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from catboost import CatBoostClassifier
import optuna
import pandas as pd
from joblib import Parallel, delayed
from functools import partial

# Desativar os logs verbosos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURAÇÃO FUNDAMENTAL
# ==============================================================================
BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST     = 'ADReSSo21_test'

# --- FUNÇÃO DE TUNAGEM (OPTUNA) ---
def objective(X_train, y_train, trial):
    """Hyperparameter optimization for CatBoost."""
    if len(X_train) < 5:
        return 0.0

    param = {
        'iterations': trial.suggest_int('iterations', 5, 120), # Leve redução no teto para acelerar os 135 trials
        'depth': trial.suggest_int('depth', 4, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False,
        'boost_from_average': True,
        'thread_count': 1  # Crucial: Mantém 1 thread por trial para não travar o loop externo paralelo
    }

    clf = CatBoostClassifier(**param)
    try:
        # Cross validation rápido de 2 folds como no seu script original
        score = cross_val_score(clf, X_train, y_train, cv=2, scoring='accuracy').mean()
        return score
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
        current_label = None
        if any(kw in folder_name for kw in ['HC', 'CONTROL', 'CN', 'HEALTHY', 'NORMAL']):
            current_label = 'HC'
        elif any(kw in folder_name for kw in ['AD', 'ALZ', 'PATIENT', 'DEMENTIA']):
            current_label = 'AD'
        else:
            for f in files[:1]:
                f_low = f.lower()
                if 'hc' in f_low or 'control' in f_low: current_label = 'HC'
                elif 'ad' in f_low or 'alz' in f_low: current_label = 'AD'

        if not current_label: continue
        for file in files:
            if file.endswith('.npy'):
                try:
                    data = np.load(os.path.join(root, file)).flatten().astype(np.float32)
                    if expected_dim is None:
                        expected_dim = data.size
                    elif data.size != expected_dim:
                        continue
                    if np.isnan(data).all():
                        continue
                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    continue
    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Worker function used by Parallel execution."""
    print(f"[>] Processing: {feature_folder}")
    X_train_full, y_train_full, _ = load_data_for_feature_set(feature_folder, train_ds)
    if len(X_train_full) < 5 or len(np.unique(y_train_full)) < 2:
        return None

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0 or X_test.shape[1] != X_train_full.shape[1]:
        return None

    # --- DEFENSOR DE ALTA DIMENSIONALIDADE (Prevenção para o ComParE_2016_6k) ---
    if X_train_full.shape[1] > 500:
        print(f"    [i] High-dim detected ({X_train_full.shape[1]} features). Applying fast filtering...")

        # 1. Filtro de variância baixa
        selector_var = VarianceThreshold(threshold=0.01)
        X_train_full = selector_var.fit_transform(X_train_full)
        X_test = selector_var.transform(X_test)

        # 2. Seleção estatística rápida ANOVA
        k_features = min(200, X_train_full.shape[1])
        selector_k = SelectKBest(score_func=f_classif, k=k_features)
        X_train_full = selector_k.fit_transform(X_train_full, y_train_full)
        X_test = selector_k.transform(X_test)
        print(f"    [i] Optimized matrix shape for CatBoost: {X_train_full.shape}")

    # Split para o Early Stopping do treinamento final
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Configuração do Estudo do Optuna focado em velocidade linear constante
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    optimized_objective = lambda trial: objective(X_train, y_train, trial)

    # Executa exatamente os 135 trials
    study.optimize(optimized_objective, n_trials=135)

    # Treinamento Final com os melhores parâmetros encontrados
    final_params = study.best_params.copy()

    if 'und_l2_reg' in final_params:
        final_params['l2_leaf_reg'] = final_params.pop('und_l2_reg')

    final_params.update({
        'random_seed': 42,
        'verbose': 0,
        'thread_count': 1, # Mantém leve no encerramento paralelo
        'allow_writing_files': False
    })

    clf = CatBoostClassifier(**final_params)
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)

    classes = clf.classes_
    try:
        ad_idx = np.where(classes == 'AD')[0][0]
    except IndexError:
        return None

    probs = clf.predict_proba(X_test)
    prob_dict, label_dict = {}, {}
    for i in range(len(test_ids)):
        p_id = test_ids[i]
        prob_dict[p_id] = probs[i][ad_idx]
        label_dict[p_id] = 1 if y_test[i] == 'AD' else 0

    print(f"    [OK] Feature {feature_folder} finalized!")
    return feature_folder, prob_dict, label_dict

def run_pipeline():
    """Main Orchestrator."""
    print("="*70)
    print("INICIANDO PIPELINE DE ENSEMBLE (CATBOOST + OPTUNA 135 TRIALS)")
    print("="*70)
    if not BASE_FEATURES_DIR.exists():
        print(f"[!] Error: Directory not found: {BASE_FEATURES_DIR}")
        return

    all_features = [f for f in os.listdir(BASE_FEATURES_DIR) if os.path.isdir(BASE_FEATURES_DIR.joinpath(f))]
    print(f"[INFO] Distributing {len(all_features)} features to CPU cores...")

    # OTIMIZAÇÃO: Alterado backend de 'threading' para 'loky' (Processos) para evitar concorrência interna do GIL
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_single_feature_set)(feat, DATASET_TRAIN, DATA_TEST)
        for feat in all_features
    )

    # AGGREGATION PHASE
    patient_probs_accumulator = {}
    true_labels_map = {}
    features_processed_count = 0
    all_test_ids = set()

    print("\n[INFO] Aggregating results...")
    for res in results:
        if res is None: continue
        feat_name, p_probs, p_labels = res
        features_processed_count += 1
        for p_id in p_probs.keys():
            all_test_ids.add(p_id)
        for p_id, prob in p_probs.items():
            patient_probs_accumulator.setdefault(p_id, []).append(prob)
        for p_id, label in p_labels.items():
            true_labels_map[p_id] = label

    # FINAL VOTING (Soft Voting)
    final_predictions, final_ground_truth = [], []
    ids_intersecao = []

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
        df_results.to_csv('/home/dani/Documentos/PEP/ensemble_results_CB.csv', index=False)
        print("[INFO] Resultados exportados com sucesso em 'ensemble_results_CB.csv'!")
    # ==========================================================================

    if not final_predictions:
        print("[!] Error: No common patients found.")
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
