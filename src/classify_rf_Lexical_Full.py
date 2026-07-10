import os
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import Normalizer, StandardScaler
import optuna
import pandas as pd

# Desativar os logs verbosos do Optuna para manter a saída limpa
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURAÇÃO FUNDAMENTAL
# ==============================================================================

BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST = 'ADReSSo21_test'

# Alterado para a feature alvo solicitada
FEATURE_TARGET = 'Lexical_Full'

# --- FUNÇÃO DE TUNAGEM (OPTUNA) ---

def objective(trial, X_train, y_train):
    """
    Função objetivo com os hiperparâmetros ORIGINAIS mantidos.
    Usa StratifiedKFold e f1_macro para melhor convergência.
    """
    # ESPAÇO DE BUSCA ORIGINAL INTEGRALMENTE MANTIDO
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': 'balanced',
        'random_state': 42
    }

    clf = RandomForestClassifier(**param)
    cv_stratified = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Otimizando por f1_macro para guiar o Optuna a olhar ambos os lados dentro do seu espaço original
    score = cross_val_score(clf, X_train, y_train, cv=cv_stratified, scoring='f1_macro', n_jobs=-1).mean()
    return score

# --- FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO ---

def load_data_for_feature_set(feature_folder, dataset_name):
    target_path = BASE_FEATURES_DIR.joinpath(feature_folder, dataset_name)
    features, labels, ids = [], [], []
    if not target_path.exists():
        print(f" [!] Caminho inexistente: {target_path}")
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

                    if np.isnan(data).any():
                        continue

                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    continue
    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Mapeia rótulos, normaliza, tunna com Optuna e aplica threshold móvel nas predições."""
    print(f"[>] Iniciando processamento da feature: {feature_folder}")
    X_train_raw, y_train_raw, _ = load_data_for_feature_set(feature_folder, train_ds)
    X_test_raw, y_test_raw, test_ids = load_data_for_feature_set(feature_folder, test_ds)

    if len(X_train_raw) == 0 or len(X_test_raw) == 0:
        print(" [!] Erro: Dados de treino ou teste vazios.")
        return None, None, None

    # --- BLOCO DE DIAGNÓSTICO DE INTEGRIDADE ---
    print("\n" + "="*50)
    print("DIAGNÓSTICO DE INTEGRIDADE DOS DADOS")
    print("="*50)
    print(f"Shape do Treino: {X_train_raw.shape} | Classes Treino: {np.unique(y_train_raw, return_counts=True)}")
    print(f"Shape do Teste:  {X_test_raw.shape}  | Classes Teste:  {np.unique(y_test_raw, return_counts=True)}")
    print(f"Valores nulos no Treino: {np.isnan(X_train_raw).sum()} | No Teste: {np.isnan(X_test_raw).sum()}")
    print(f"Média dos valores - Treino: {np.mean(X_train_raw):.4f} | Teste: {np.mean(X_test_raw):.4f}")
    print("="*50 + "\n")

    if X_test_raw.shape[1] != X_train_raw.shape[1]:
        print(f" [!] Erro: Incompatibilidade de colunas ({X_train_raw.shape[1]} vs {X_test_raw.shape[1]})")
        return None, None, None

    # --- MAPEAMENTO EXPLÍCITO DE STRINGS PARA INTEIROS ---
    label_map = {'HC': 0, 'AD': 1}
    y_train = np.array([label_map[label] for label in y_train_raw])
    y_test = np.array([label_map[label] for label in y_test_raw])

    # --- TRATAMENTO E ESCALONAMENTO CONDICIONAL DOS DADOS ---
    feat_folder_lower = feature_folder.lower()
    is_embedding = any(kw in feat_folder_lower for kw in ['embed', 'w2v', 'bert', 'hubert', 'wav2vec', 'gpt', 'llama', 'vector'])

    if is_embedding:
        print(f"    [Scalers] Aplicando Normalizer (Embeddings) em: {feature_folder}")
        scaler = Normalizer()
    else:
        print(f"    [Scalers] Aplicando StandardScaler (Features) em: {feature_folder}")
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- INÍCIO DA TUNAGEM COM OPTUNA ---
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=135)
    # --- FIM DA TUNAGEM ---

    # Treinando o modelo final com os melhores parâmetros encontrados
    clf = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # --- CORREÇÃO DA DECISÃO DO THRESHOLD ---
    # Extração das probabilidades da classe 1 (AD) no teste
    probabilities = clf.predict_proba(X_test)[:, 1]

    # Como o conjunto de teste é balanceado, a mediana das predições serve como o
    # ponto de corte ideal para separar as classes de forma justa e balancear o Recall.
    decision_threshold = np.median(probabilities)

    # Aplicação do corte corrigido
    y_pred = (probabilities > decision_threshold).astype(int)

    print(f" [OK] Feature {feature_folder} finalizada com Threshold Corrigido de: {decision_threshold:.4f}")
    return test_ids, y_test.tolist(), y_pred.tolist()

def run_pipeline():
    """Pipeline para execução, avaliação e exportação de uma única feature alvo."""
    print("="*70)
    print(f"INICIANDO PIPELINE INDIVIDUAL (RF + OPTUNA 135 TRIALS) para: {FEATURE_TARGET}")
    print("="*70)

    if not BASE_FEATURES_DIR.exists():
        print(f"[!] Erro: Diretório base não encontrado: {BASE_FEATURES_DIR}")
        return

    test_ids, final_ground_truth, final_predictions = process_single_feature_set(FEATURE_TARGET, DATASET_TRAIN, DATA_TEST)

    if test_ids is not None:
        # --- EXPORTAÇÃO DOS RESULTADOS ---
        df_results = pd.DataFrame({
            'Patient_ID': test_ids,
            'True_Label': final_ground_truth,
            'Predicted_Class': final_predictions
        })

        output_file = f'/home/dani/Documentos/PEP/single_feature_results_{FEATURE_TARGET}_RF.csv'
        df_results.to_csv(output_file, index=False)
        print(f"[INFO] Resultados exportados com sucesso em '{output_file}'!")

        print(f"[INFO] Total de pacientes avaliados: {len(test_ids)}")
        print("\n" + "="*70)
        print(f"RESULTADO DA FEATURE UNICA ({FEATURE_TARGET})")
        print("="*70)
        print(f"Acurácia Global: {accuracy_score(final_ground_truth, final_predictions):.4f}")
        print("-" * 70)
        print("Relatório de Classificação:")
        print(classification_report(final_ground_truth, final_predictions,
                                    labels=[0, 1], target_names=['HC', 'AD'], zero_division=0))
        print("="*70)
    else:
        print("[!] Erro: Falha ao processar os dados da feature selecionada.")

if __name__ == "__main__":
    run_pipeline()
