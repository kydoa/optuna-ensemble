import os
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif # Otimizadores de matriz
import optuna
import pandas as pd

# Importações necessárias para o paralelismo de processos
from concurrent.futures import ProcessPoolExecutor, as_completed

# Desativar os logs verbosos do Optuna para manter a saída paralela limpa
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
    Sugere hiperparâmetros e avalia a performance usando Cross-Validation rápido.
    """
    param = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 10), # Reduzido de 32 para 10 para evitar loops infinitos em 6k
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': 'balanced',
        'random_state': 42
    }

    clf = DecisionTreeClassifier(**param)

    # Reduzido de cv=5 para cv=3 para ganho de velocidade linear massivo
    score = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()
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
        elif any(kw in folder_name for kw in ['AD', 'ALZ', 'PATIENT', 'DEMENTIA']):
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
    """Treina com Optuna e extrai probabilidades do teste com barreira de alta dim."""
    print(f"[>] Iniciando processamento da feature: {feature_folder}")
    X_train, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print(f"    [!] Erro: Sem dados suficientes para {feature_folder}")
        return {}, {}

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0: return {}, {}

    # --- DEFENSOR DE PERFORMANCE (Filtro Ultrarápido para o ComParE_2016_6k) ---
    if X_train.shape[1] > 500:
        print(f"    [i] Alta dimensão detectada ({X_train.shape[1]} colunas). Filtrando matriz...")

        # 1. Remove colunas de variância nula/estática
        selector_var = VarianceThreshold(threshold=0.01)
        X_train = selector_var.fit_transform(X_train)
        X_test = selector_var.transform(X_test)

        # 2. Seleção ANOVA estatística instantânea para entregar 100 features limpas para a Árvore
        k_features = min(100, X_train.shape[1])
        selector_k = SelectKBest(score_func=f_classif, k=k_features)
        X_train = selector_k.fit_transform(X_train, y_train)
        X_test = selector_k.transform(X_test)
        print(f"    [i] Matriz otimizada para a Árvore de Decisão: {X_train.shape}")

    # --- INÍCIO DA TUNAGEM COM OPTUNA ---
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=135)
    # --- FIM DA TUNAGEM ---

    # Treinando o modelo final com os melhores parâmetros encontrados
    clf = DecisionTreeClassifier(**study.best_params, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    classes = clf.classes_
    try:
        ad_idx = np.where(classes == 'AD')[0][0]
    except IndexError:
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
    print("INICIANDO PIPELINE DE ENSEMBLE (PARALELIZADO: DT + OPTUNA 135 TRIALS)")
    print("="*70)

    if not BASE_FEATURES_DIR.exists():
        print(f"[!] Erro: Diretório base não encontrado: {BASE_FEATURES_DIR}")
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

    # --- EXECUÇÃO EM PARALELO POR MULTIPROCESSAMENTO ---
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = {
            executor.submit(process_single_feature_set, feat, DATASET_TRAIN, DATA_TEST): feat
            for feat in all_features
        }

        for future in as_completed(futures):
            feat = futures[future]
            try:
                p_probs, p_labels = future.result()
                if not p_probs: continue

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

    # --- SALVAR RESULTADOS PARA GRAFICOS NO JUPYTER ---
    if final_predictions:
        df_results = pd.DataFrame({
            'Patient_ID': ids_intersecao,
            'True_Label': final_ground_truth,
            'Predicted_Class': final_predictions
        })
        df_results.to_csv('/home/dani/Documentos/PEP/ensemble_results_dt.csv', index=False)
        print("[INFO] Resultados exportados com sucesso em 'ensemble_results_dt.csv'!")

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
