import os
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
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
FEATURE_TARGET = 'pAA'

# --- FUNÇÃO DE TUNAGEM (OPTUNA) ---

def objective(trial, X_train, y_train):
    """
    Função objetivo para o Optuna.
    Sugere hiperparâmetros para o SVC e avalia via Cross-Validation.
    """
    param = {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'class_weight': 'balanced',
        'random_state': 42
    }

    # Se o kernel for polinomial, sugere o grau (degree)
    if param['kernel'] == 'poly':
        param['degree'] = trial.suggest_int('degree', 2, 4)

    clf = SVC(**param)

    # Executa o Cross-Validation com CV=4 buscando otimizar a ACCURACY.
    # O paralelismo roda nas threads do cross_val_score (n_jobs=-1).
    score = cross_val_score(clf, X_train, y_train, cv=4, scoring='accuracy', n_jobs=-1).mean()
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

                    # CORREÇÃO: Se houver NaN, o dado é simplesmente ignorado.
                    if np.isnan(data).any():
                        continue

                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    continue
    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Normaliza, treina com Optuna (SVC) e extrai predições do teste para exportação."""
    print(f"[>] Iniciando processamento da feature: {feature_folder}")
    X_train, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print(f" [!] Erro: Sem dados suficientes para {feature_folder}")
        return None, None, None

    X_test, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test) == 0 or X_test.shape[1] != X_train.shape[1]:
        print(f" [!] Erro: Dados de teste ausentes ou incompatíveis para {feature_folder}")
        return None, None, None

    # --- TRATAMENTO E ESCALONAMENTO CONDICIONAL DOS DADOS ---
    feat_folder_lower = feature_folder.lower()
    is_embedding = any(kw in feat_folder_lower for kw in ['embed', 'w2v', 'bert', 'hubert', 'wav2vec', 'gpt', 'llama', 'vector', 'vggish', 'trill'])

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

    # Treinando o modelo final com os melhores parâmetros encontrados
    clf = SVC(**study.best_params, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    # Realizando as predições no conjunto de teste
    y_pred = clf.predict(X_test)

    # Conversão das predições textuais em binárias para salvar no CSV no padrão do seu pipeline original
    pred_bin = [1 if pred == 'AD' else 0 for pred in y_pred]
    true_bin = [1 if true == 'AD' else 0 for true in y_test]

    print(f" [OK] Feature {feature_folder} finalizada!")
    return test_ids, true_bin, pred_bin

def run_pipeline():
    """Pipeline para execução, avaliação e exportação de uma única feature alvo."""
    print("="*70)
    print(f"INICIANDO PIPELINE INDIVIDUAL (SVC + OPTUNA 135 TRIALS) para: {FEATURE_TARGET}")
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

        # Salvando os resultados com o sufixo _SVC indicando o modelo correto
        output_file = f'/home/dani/Documentos/PEP/single_feature_results_{FEATURE_TARGET}_SVC.csv'
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
