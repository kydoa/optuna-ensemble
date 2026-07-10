import os
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer, StandardScaler  # IMPORTANTE: Adicionado StandardScaler
import optuna
import pandas as pd

# 1. Importação necessária para o paralelismo
from concurrent.futures import ProcessPoolExecutor, as_completed

# Desativar os logs verbosos do Optuna para não poluir o terminal em paralelo
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURAÇÃO FUNDAMENTAL
# ==============================================================================
BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST     = 'ADReSSo21_test'

# --- FUNÇÃO DE TUNAGEM (OPTUNA) ---

def objective(trial, X_train, y_train):
    """
    Função objetivo para o Optuna.
    Sugere hiperparâmetros para o Random Forest e avalia via Cross-Validation.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': 1  # Mantido em 1 para não colidir com o ProcessPoolExecutor externo
    }

    clf = RandomForestClassifier(**param)

    # AJUSTADO: cv=4 e scoring='accuracy' para encontrar os melhores hiperparâmetros
    score = cross_val_score(clf, X_train, y_train, cv=4, scoring='accuracy', n_jobs=1).mean()
    return score

# --- FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO ---

def load_data_for_feature_set(feature_folder, dataset_name):
    target_path = BASE_FEATURES_DIR.joinpath(feature_folder, dataset_name)
    features, labels, ids = [], [], []
    if not target_path.exists():
        print(f"    [!] Caminho inexistente: {target_path}")
        return np.array([]), [], []

    for root, dirs, files in os.walk(target_path):
        if not files: continue
        folder_name = Path(root).name.upper()
        current_label = None

        if any(kw in folder_name for kw in ['HC', 'CONTROL', 'CN', 'HEALTHY', 'NORMAL']):
            current_label = 'HC'
        elif any(lan in folder_name for lan in ['AD', 'ALZ', 'PATIENT', 'DEMENTIA']):
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

                    # Se houver qualquer dado nulo (NaN), ignora a amostra (Sem Imputer)
                    if np.isnan(data).any():
                        continue

                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    continue

    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Normaliza embeddings/features, treina com Tunagem Optuna e extrai probabilidades do teste."""
    print(f"[>] Iniciando processamento da feature: {feature_folder}")
    X_train, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print(f"    [!] Erro: Sem dados suficientes para {feature_folder}")
        return {}, {}

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0 or X_test.shape[1] != X_train.shape[1]:
        return {}, {}

    # --- TRATAMENTO E ESCALONAMENTO CONDICIONAL DOS DADOS ---
    # Heurística para detectar se a pasta refere-se a embeddings
    feat_folder_lower = feature_folder.lower()
    is_embedding = any(kw in feat_folder_lower for kw in ['embed', 'w2v', 'bert', 'hubert', 'wav2vec', 'gpt', 'llama', 'vector'])

    if is_embedding:
        print(f"    [Scalers] Aplicando Normalizer (Embeddings) em: {feature_folder}")
        scaler = Normalizer()
    else:
        print(f"    [Scalers] Aplicando StandardScaler (Features) em: {feature_folder}")
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- INÍCIO DA TUNAGEM COM OPTUNA ---
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=135)
    # --- FIM DA TUNAGEM ---

    # Criamos o modelo final usando os melhores parâmetros encontrados pelo Optuna
    clf = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)

    classes = clf.classes_
    try:
        ad_idx = list(classes).index('AD')
    except (IndexError, ValueError):
        print(f"    [!] Aviso: Classe 'AD' não encontrada em {feature_folder}.")
        return {}, {}

    probs = clf.predict_proba(X_test)
    prob_dict = {}
    label_dict = {}

    for i in range(len(test_ids)):
        p_id = test_ids[i]
        prob_dict[p_id] = probs[i][ad_idx]
        label_dict[p_id] = 1 if y_test[i] == 'AD' else 0

    print(f"    [OK] Feature {feature_folder} finalizada!")
    return prob_dict, label_dict

def run_pipeline():
    """Pipeline de Soft Voting Ensemble Paralelizado."""
    print("="*70)
    print("INICIANDO PIPELINE DE ENSEMBLE (PARALELIZADO: SOFT VOTING + OPTUNA 135 TRIALS)")
    print("="*70)

    if not BASE_FEATURES_DIR.exists():
        print(f"[!] Diretório base não existe: {BASE_FEATURES_DIR}")
        return

    all_features = [f for f in os.listdir(BASE_FEATURES_DIR)
                    if os.path.isdir(BASE_FEATURES_DIR.joinpath(f))]

    if not all_features:
        print("[!] Nenhuma pasta de feature encontrada.")
        return

    patient_probs_accumulator = {}
    true_labels_map = {}
    features_processed_count = 0
    all_test_ids = set()

    # --- BLOCO DE EXECUÇÃO EM PARALELO ---
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = {
            executor.submit(process_single_feature_set, feat, DATASET_TRAIN, DATA_TEST): feat
            for feat in all_features
        }

        for future in as_completed(futures):
            feat = futures[future]
            try:
                p_probs, p_labels = future.result()

                if not p_probs:
                    continue

                for p_id in p_probs.keys():
                    all_test_ids.add(p_id)
                for p_id, prob in p_probs.items():
                    patient_probs_accumulator.setdefault(p_id, []).append(prob)
                for p_id, label in p_labels.items():
                    true_labels_map[p_id] = label
                features_processed_count += 1

            except Exception as e:
                print(f"[!] Erro crítico ao processar a feature '{feat}': {e}")

    # --- FIM DO PARALELISMO ---

    print("\n" + "="*70)
    print("FASE FINAL: ALINHAMENTO E MÉDIA (SOFT VOTING)")
    print("="*70)

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

    if final_predictions:
        df_results = pd.DataFrame({
            'Patient_ID': ids_intersecao,
            'True_Label': final_ground_truth,
            'Predicted_Class': final_predictions
        })
        df_results.to_csv('/home/dani/Documentos/PEP/ensemble_results_RF.csv', index=False)
        print("[INFO] Resultados exportados com sucesso em 'ensemble_results_RF.csv'!")

    if not final_predictions:
        print("[!] Erro: Nenhum paciente comum encontrado.")
        return

    print(f"[INFO] Pacientes na interseção total: {len(ids_intersecao)}")
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
