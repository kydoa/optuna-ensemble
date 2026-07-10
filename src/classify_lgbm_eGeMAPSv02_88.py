import os
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from lightgbm import LGBMClassifier
import optuna
import pandas as pd

# Desativar os logs verbosos do Optuna e do LightGBM
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURAÇÃO FUNDAMENTAL
# ==============================================================================
BASE_FEATURES_DIR = Path('/home/dani/Documentos/PEP/admodel-master-version/data/features')
DATASET_TRAIN = 'ADReSSo21_train'
DATA_TEST = 'ADReSSo21_test'

# Alterado para a feature alvo solicitada
FEATURE_TARGET = 'eGeMAPSv02_88'

# --- FUNÇÃO DE TUNAGEM (OPTUNA) ---
def objective(trial, X_train, y_train_encoded):
    """Otimização de hiperparâmetros para o LightGBM."""
    if len(X_train) < 5:
        return 0.0

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': 1  # 1 thread por modelo interno durante a busca do Optuna
    }

    # Tratamento de desbalanceamento de classes no LightGBM
    neg_count = np.sum(y_train_encoded == 0)
    pos_count = np.sum(y_train_encoded == 1)
    if pos_count > 0:
        param['scale_pos_weight'] = neg_count / pos_count

    clf = LGBMClassifier(**param)
    try:
        score = cross_val_score(clf, X_train, y_train_encoded, cv=4, scoring='accuracy', n_jobs=-1).mean()
        return score
    except Exception:
        return 0.0

# --- FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO ---
def load_data_for_feature_set(feature_folder, dataset_name):
    target_path = BASE_FEATURES_DIR.joinpath(feature_folder, dataset_name)
    features, labels, ids = [], [], []
    if not target_path.exists():
        print(f" [!] Caminho inexistente: {target_path}")
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
                    if np.isnan(data).any():
                        continue
                    features.append(data)
                    ids.append(Path(file).stem)
                    labels.append(current_label)
                except Exception:
                    continue
    return np.array(features), np.array(labels), ids

def process_single_feature_set(feature_folder, train_ds, test_ds):
    """Trata dados, otimiza via Optuna e infere predições com o LightGBM."""
    print(f"[>] Iniciando processamento da feature: {feature_folder}")
    X_train_raw, y_train, _ = load_data_for_feature_set(feature_folder, train_ds)
    if len(X_train_raw) < 5 or len(np.unique(y_train)) < 2:
        print(f" [!] Erro: Sem dados suficientes para {feature_folder}")
        return None, None, None

    X_test_raw, y_test, test_ids = load_data_for_feature_set(feature_folder, test_ds)
    if len(X_test_raw) == 0 or X_test_raw.shape[1] != X_train_raw.shape[1]:
        print(f" [!] Erro: Dados de teste ausentes ou incompatíveis para {feature_folder}")
        return None, None, None

    # --- CODIFICAÇÃO MANDATÓRIA DE RÓTULOS ---
    label_map = {'HC': 0, 'AD': 1}
    y_train_encoded = np.array([label_map[label] for label in y_train])
    y_test_encoded = np.array([label_map[label] for label in y_test])

    # --- TRATAMENTO E ESCALONAMENTO CONDICIONAL ---
    feat_folder_lower = feature_folder.lower()
    is_embedding = any(kw in feat_folder_lower for kw in ['embed', 'w2v', 'bert', 'hubert', 'wav2vec', 'gpt', 'llama', 'vector', 'vggish', 'trill'])

    if is_embedding:
        print(f"    [Scalers] Aplicando Normalizer (Embeddings) em: {feature_folder}")
        scaler = Normalizer()
    else:
        print(f"    [Scalers] Aplicando StandardScaler (Features) em: {feature_folder}")
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Conversão para DataFrame com mapeamento explícito de colunas de texto para o LightGBM
    feature_names = [f"feat_{i}" for i in range(X_train_scaled.shape[1])]
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Split interno estruturado para validação do Early Stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_df, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
    )

    # Configuração do Estudo do Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    study.optimize(lambda trial: objective(trial, X_train_df, y_train_encoded), n_trials=135)

    # Configurando os parâmetros finais encontrados
    final_params = study.best_params.copy()
    final_params.update({
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    })

    # Treinamento final associando os callbacks de Early Stopping
    clf = LGBMClassifier(**final_params)
    from lightgbm import early_stopping, log_evaluation
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=20, verbose=False), log_evaluation(0)]
    )

    # Realizando as predições no conjunto de teste
    y_pred = clf.predict(X_test_df)

    pred_bin = y_pred.tolist()
    true_bin = y_test_encoded.tolist()

    print(f" [OK] Feature {feature_folder} finalizada!")
    return test_ids, true_bin, pred_bin

def run_pipeline():
    """Pipeline para execução, avaliação e exportação de uma única feature alvo."""
    print("="*70)
    print(f"INICIANDO PIPELINE INDIVIDUAL (LightGBM + OPTUNA 135 TRIALS) para: {FEATURE_TARGET}")
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

        # Nome do arquivo final salvo com o sufixo _LGBM
        output_file = f'/home/dani/Documentos/PEP/single_feature_results_{FEATURE_TARGET}_LGBM.csv'
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
