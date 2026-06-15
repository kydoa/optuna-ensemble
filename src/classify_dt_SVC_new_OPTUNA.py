import os
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif # Otimizadores de matriz
import optuna
# Import for parallel processing
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Desativar os logs verbosos do Optuna para manter a saída paralela organizada
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST     = 'ADReSSo21_test'

# --- TUNING FUNCTION (OPTUNA) ---
def objective(trial, X_train, y_train):
    param = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        # Removido 'poly' do espaço de busca pois gera loops infinitos em matrizes de áudio complexas
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'sigmoid']),
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42
    }

    # O Pipeline interno lida apenas com o classificador pós-filtrado
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svc', SVC(**param))
    ])

    # Reduzido de cv=5 para cv=3 para viabilizar as 135 iterações sem travar a CPU
    score = cross_val_score(model_pipeline, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

# --- DATA LOADING AND PROCESSING ---
def load_data_for_feature_set(feature_folder, dataset_name):
    target_path = BASE_FEATURES_DIR.joinpath(feature_folder, dataset_name)
    features, labels, ids = [], [], []
    if not target_path.exists():
        return np.array([]), [], []

    for root, dirs, files in os.walk(target_path):
        if not files: continue
        folder_name = Path(root).name.upper()
        current_label = None

        if any(kw in folder_name for kw in ['HC', 'CONTROL', 'CN', 'HEALTHY', 'NORMAL']):
            current_label = 'HC'
        elif any(kw in folder_name for kw in ['AD', 'AL_PATIENT', 'DEMENTIA', 'ALZ']):
            current_label = 'AD'
        else:
            for f in files[:1]:
                if 'hc' in f.lower() or 'control' in f.lower():
                    current_label = 'HC'
                elif 'ad' in f.lower() or 'alz' in f.lower():
                    current_label = 'AD'

        if not current_label: continue

        for file in files:
            if file.endswith('.npy'):
                try:
                    data = np.load(os.path.join(root, file)).flatten().astype(np.float32)
                    if np.isnan(data).all(): continue
                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    continue
    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    print(f"[>] Processing feature: {feature_folder}")
    X_train, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)
    if len(X_train) == 0 or len(np.unique(y_train)) < 2: return None

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0: return None

    # Preprocessing setup inicial
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train_transformed = imputer.fit_transform(X_train)
    X_train_transformed = scaler.fit_transform(X_train_transformed)

    X_test_transformed = imputer.transform(X_test)
    X_test_transformed = scaler.transform(X_test_transformed)

    # --- DEFENSOR DE PERFORMANCE (Barreira Física para os 135 Trials no ComParE_2016_6k) ---
    if X_train_transformed.shape[1] > 500:
        print(f"    [i] Alta dimensão detectada no SVC ({X_train_transformed.shape[1]} colunas). Filtrando...")

        selector_var = VarianceThreshold(threshold=0.01)
        X_train_transformed = selector_var.fit_transform(X_train_transformed)
        X_test_transformed = selector_var.transform(X_test_transformed)

        k_features = min(100, X_train_transformed.shape[1])
        selector_k = SelectKBest(score_func=f_classif, k=k_features)
        X_train_transformed = selector_k.fit_transform(X_train_transformed, y_train)
        X_test_transformed = selector_k.transform(X_test_transformed)
        print(f"    [i] Matriz reduzida para o SVC: {X_train_transformed.shape}")

    # --- INÍCIO DA TUNAGEM COM OPTUNA ---
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    study.optimize(lambda trial: objective(trial, X_train_transformed, y_train), n_trials=135)
    # --- FIM DA TUNAGEM ---

    # Treinamento Final
    clf = SVC(**study.best_params, probability=True, class_weight='balanced', random_state=42)
    clf.fit(X_train_transformed, y_train)

    try:
        ad_idx = np.where(clf.classes_ == 'AD')[0][0]
    except IndexError:
        print(f"[!] Error: 'AD' class not found in model classes for {feature_folder}")
        return None

    probs = clf.predict_proba(X_test_transformed)

    prob_dict = {}
    label_dict = {}
    for i in range(len(test_ids)):
        p_id = test_ids[i]
        prob_dict[p_id] = probs[i][ad_idx]
        label_dict[p_id] = 1 if y_test[i] == 'AD' else 0

    print(f"    [OK] Feature {feature_folder} finalized!")
    return feature_folder, prob_dict, label_dict

def run_pipeline():
    print("="*70)
    print("STARTING SVC ENSEMBLE PIPELINE (PARALLEL + SOFT VOTING + OPTUNA 135 TRIALS)")
    print("="*70)

    all_features = [f for f in os.listdir(BASE_FEATURES_DIR)
                    if os.path.isdir(BASE_FEATURES_DIR.joinpath(f))]

    if not all_features:
        print("[!] Nenhuma pasta de feature encontrada.")
        return

    patient_probs_accumulator = {}
    true_labels_map = {}
    features_processed_count = 0
    all_test_ids = set()

    # Parallel Execution
    print(f"[INFO] Distributing tasks to CPU cores for {len(all_features)} features...")
    results = Parallel(n_jobs=-1)(
        delayed(process_single_feature_set)(feat, DATASET_TRAIN, DATA_TEST)
        for feat in all_features
    )

    # Aggregation
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

    # Alignment and Voting
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
    # GRAVAÇÃO DO DATAFRAME COMPATÍVEL COM O SEU JUPYTER NOTEBOOK
    # ==========================================================================
    if final_predictions:
        df_results = pd.DataFrame({
            'Patient_ID': ids_intersecao,
            'True_Label': final_ground_truth,
            'Predicted_Class': final_predictions
        })
        df_results.to_csv('/home/dani/Documentos/PEP/ensemble_results_SVC.csv', index=False)
        print("[INFO] Resultados exportados com sucesso em 'ensemble_results_SVC.csv'!")
    # ==========================================================================

    if not final_predictions:
        print("[!] Error: No common patients found across feature sets.")
        return

    print(f"[INFO] Pacientes na interseção total: {len(ids_intersecao)}")
    print("\n" + "="*70)
    print("RESULT OF ENSEMBLE (Soft Voting)")
    print("="*70)
    print(f"Global Accuracy: {accuracy_score(final_ground_truth, final_predictions):.4f}")
    print("-" * 70)
    print("Classification Report:")
    print(classification_report(final_ground_truth, final_predictions,
                                labels=[0, 1], target_names=['HC', 'AD'], zero_division=0))
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
