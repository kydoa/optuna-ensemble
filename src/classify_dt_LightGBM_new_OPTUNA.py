import os
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import lightgbm as lgb
import optuna
import pandas as pd
from joblib import Parallel, delayed

# Desativar os logs detalhados do Optuna para manter o terminal limpo e legível
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURAÇÃO FUNDAMENTAL
# ==============================================================================
BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST     = 'ADReSSo21_test'

def load_data_for_feature_set(feature_folder, dataset_name):
    target_path = BASE_FEATURES_DIR.joinpath(feature_folder, dataset_name)
    features, labels, ids = [], [], []

    if not target_path.exists():
        print(f"    [!] Caminho inexistente: {target_path}")
        return np.array([]), [], []

    skipped = 0

    for root, dirs, files in os.walk(target_path):
        if not files:
            continue

        folder_name = Path(root).name.upper()
        current_label = None

        if any(kw in folder_name for kw in ['HC', 'CONTROL', 'CN', 'HEALTHY', 'NORMAL']):
            current_label = 'HC'
        elif any(kw in folder_name for kw in ['AD', 'ALZ', 'PATIENT', 'DEMENTIA']):
            current_label = 'AD'
        else:
            for f in files[:1]:
                if 'hc' in f.lower() or 'control' in f.lower():
                    current_label = 'HC'
                elif 'ad' in f.lower() or 'alz' in f.lower():
                    current_label = 'AD'

        if not current_label:
            continue

        for file in files:
            if file.endswith('.npy'):
                try:
                    data = np.load(os.path.join(root, file)).flatten()
                    if len(data) == 0 or np.isnan(data).all():
                        skipped += 1
                        continue
                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    skipped += 1

    return np.array(features), np.array(labels), ids


def optimize_hyperparameters(X_train, y_train):
    """Encontra os melhores hiperparâmetros usando Optuna e Cross-Validation."""

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42,
            'class_weight': 'balanced',
            'verbosity': -1,
            # 🏎️ 1 thread por classificador individual para viabilizar o paralelismo externo multi-pasta
            'n_jobs': 1,

            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 15),
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            # Mapeamento explícito de strings para int exigido pelo LightGBM nativo
            y_tr_mapped = np.where(y_tr == 'AD', 1, 0)
            y_va_mapped = np.where(y_va == 'AD', 1, 0)

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr_mapped)

            preds = clf.predict(X_va)
            scores.append(accuracy_score(y_va_mapped, preds))

        return np.mean(scores)

    # Motor de busca linear ultraleve com Pruner agressivo para gerenciar perfeitamente os 135 trials
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    # Executa os 135 trials solicitados
    study.optimize(objective, n_trials=135, n_jobs=1)

    return study.best_params


def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Worker encapsulado para processamento paralelo via Joblib."""
    print(f"[>] Processing: {feature_folder}")

    X_train, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return None

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0 or X_test.shape[1] != X_train.shape[1]:
        return None

    # --- DEFENSOR DE ALTA DIMENSIONALIDADE (Prevenção física para o ComParE_2016_6k) ---
    if X_train.shape[1] > 500:
        print(f"    [i] High-dim detected ({X_train.shape[1]} features) on {feature_folder}. Applying fast filtering...")

        # 1. Filtro rápido de variância irrelevante
        selector_var = VarianceThreshold(threshold=0.01)
        X_train = selector_var.fit_transform(X_train)
        X_test = selector_var.transform(X_test)

        # 2. Seleção ANOVA estatística para reduzir a carga drástica do LightGBM para 200 colunas ótimas
        k_features = min(200, X_train.shape[1])
        selector_k = SelectKBest(score_func=f_classif, k=k_features)
        X_train = selector_k.fit_transform(X_train, y_train)
        X_test = selector_k.transform(X_test)
        print(f"    [i] Optimized matrix shape for LightGBM: {X_train.shape}")

    best_params = optimize_hyperparameters(X_train, y_train)

    # Atribui threads seguras para o fit de encerramento
    best_params['n_jobs'] = 1
    best_params['verbosity'] = -1

    # Mapeamento do treino completo antes do fit final do modelo
    y_train_mapped = np.where(y_train == 'AD', 1, 0)

    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X_train, y_train_mapped)

    probs = clf.predict_proba(X_test)

    # LightGBM mapeia internamente as classes ordenadas ([0, 1] mapeia para [0=HC, 1=AD])
    ad_idx = 1

    prob_dict = {}
    label_dict = {}
    for i in range(len(test_ids)):
        p_id = test_ids[i]
        prob_dict[p_id] = probs[i][ad_idx]
        label_dict[p_id] = 1 if y_test[i] == 'AD' else 0

    print(f"    [OK] Feature {feature_folder} finalized!")
    return feature_folder, prob_dict, label_dict


def run_pipeline():
    """Pipeline de Soft Voting Ensemble rodando em paralelo assíncrono."""
    print("="*70)
    print("INICIANDO PIPELINE DE ENSEMBLE (LIGHTGBM + OPTUNA 135 TRIALS)")
    print("="*70)

    if not BASE_FEATURES_DIR.exists():
        print(f"[!] Erro: Diretório base não encontrado: {BASE_FEATURES_DIR}")
        return

    all_features = [f for f in os.listdir(BASE_FEATURES_DIR)
                    if os.path.isdir(BASE_FEATURES_DIR.joinpath(f))]
    if not all_features:
        print("[!] Nenhuma pasta de feature encontrada.")
        return

    print(f"[INFO] Distributing {len(all_features)} features to CPU cores...")

    # OTIMIZAÇÃO: Alterado backend para 'loky' (multiprocessing de isolamento real) para evitar gargalos com C++/OpenMP
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_single_feature_set)(feat, DATASET_TRAIN, DATA_TEST)
        for feat in all_features
    )

    # --- FASE DE AGREGAÇÃO ---
    patient_probs_accumulator = {}
    true_labels_map = {}
    features_processed_count = 0
    all_test_ids = set()

    print("\n[INFO] Aggregating results...")
    for res in results:
        if res is None:
            continue
        feat_name, p_probs, p_labels = res
        features_processed_count += 1

        for p_id in p_probs.keys():
            all_test_ids.add(p_id)
        for p_id, prob in p_probs.items():
            patient_probs_accumulator.setdefault(p_id, []).append(prob)
        for p_id, label in p_labels.items():
            true_labels_map[p_id] = label

    print("\n" + "="*70)
    print("FASE FINAL: ALINHAMENTO E MÉDIA (SOFT VOTING)")
    print("="*70)

    final_predictions = []
    final_ground_truth = []
    ids_intersecao = []
    ids_omitidos = []

    for p_id in all_test_ids:
        if p_id not in patient_probs_accumulator:
            continue

        probs_list = patient_probs_accumulator[p_id]

        if len(probs_list) == features_processed_count:
            avg_prob_ad = np.mean(probs_list)
            pred_bin = 1 if avg_prob_ad > 0.5 else 0
            true_val = true_labels_map.get(p_id, -1)

            if true_val != -1:
                final_predictions.append(pred_bin)
                final_ground_truth.append(true_val)
                ids_intersecao.append(p_id)
        else:
            ids_omitidos.append(p_id)

    # ==========================================================================
    # GRAVAÇÃO E EXPORTAÇÃO PARA O SEU JUPYTER NOTEBOOK
    # ==========================================================================
    if final_predictions:
        df_results = pd.DataFrame({
            'Patient_ID': ids_intersecao,
            'True_Label': final_ground_truth,
            'Predicted_Class': final_predictions
        })
        df_results.to_csv('/home/dani/Documentos/PEP/ensemble_results_GBM.csv', index=False)
        print("[INFO] Resultados exportados com sucesso em 'ensemble_results_GBM.csv'!")
    # ==========================================================================

    print(f"[INFO] Pacientes na interseção total: {len(ids_intersecao)}")
    if ids_omitidos:
        print(f"[ALERT] Pacientes omitidos por falta em alguma feature: {len(ids_omitidos)}")

    count_evaluated = len(final_predictions)
    if count_evaluated == 0:
        print("[!] Erro: Nenhum paciente comum encontrado em todas as pastas.")
        return

    print("\n" + "="*70)
    print("RESULTADO DO ENSEMBLE (Soft Voting)")
    print("="*70)
    print(f"Acurácia Global do Ensemble: {accuracy_score(final_ground_truth, final_predictions):.4f}")
    print("-" * 70)
    print("Relatório de Classificação:")
    print(classification_report(final_ground_truth, final_predictions,
                                labels=[0, 1], target_names=['HC', 'AD'], zero_division=0))
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
