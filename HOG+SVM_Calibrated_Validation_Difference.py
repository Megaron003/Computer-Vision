import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from skimage.feature import hog
import joblib
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json
import datetime
from collections import Counter

class HOG_SVM_CompleteValidator:
    """Validador Completo com Matriz de Confusão para Dataset de Teste"""
    
    def __init__(self, hog_parameters=None):
        if hog_parameters is None:
            self.hog_parameters = {
                'orientations': 9,
                'pixels_per_cell': (8, 8),
                'cells_per_block': (2, 2),
                'block_norm': 'L2-Hys',
                'transform_sqrt': True,
                'feature_vector': True
            }
        else:
            self.hog_parameters = hog_parameters
            
        self.svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.results = {}
    
    def check_test_structure(self, test_path):
        """Verifica se o dataset de teste está estruturado corretamente"""
        print("=" * 80)
        print("🔍 VERIFICANDO ESTRUTURA DO DATASET DE TESTE")
        print("=" * 80)
        
        expected_folders = ['Cancer', 'Healthy', 'cancer', 'healthy', 'CANCER', 'HEALTHY',
                           'Cancer_brain', 'Healthy_brain', 'cancer_brain', 'healthy_brain']
        
        found_folders = []
        cancer_folder = None
        healthy_folder = None
        
        # Verificar subpastas
        for item in os.listdir(test_path):
            item_path = os.path.join(test_path, item)
            if os.path.isdir(item_path):
                found_folders.append(item)
                
                # Tentar identificar automaticamente
                item_lower = item.lower()
                if any(keyword in item_lower for keyword in ['cancer', 'tumor', 'positive', 'malignant', 'ca', 'not']):
                    cancer_folder = item
                    print(f"✅ Identificada pasta de CÂNCER: {item}")
                elif any(keyword in item_lower for keyword in ['healthy', 'normal', 'negative', 'control', 'saudavel']):
                    healthy_folder = item
                    print(f"✅ Identificada pasta de SAUDÁVEL: {item}")
        
        # Se não identificou automaticamente
        if cancer_folder is None and len(found_folders) >= 2:
            cancer_folder = found_folders[0]
            healthy_folder = found_folders[1]
            print(f"⚠️  Assumindo: {cancer_folder} = CÂNCER, {healthy_folder} = SAUDÁVEL")
        
        if cancer_folder is None or healthy_folder is None:
            print("\n❌ ESTRUTURA INCORRETA DO DATASET DE TESTE")
            print("Por favor, organize seu dataset de teste assim:")
            print("Test_Images_Total/")
            print("├── Cancer/           # Todas imagens com câncer")
            print("│   ├── img1.jpg")
            print("│   └── ...")
            print("└── Healthy/          # Todas imagens saudáveis")
            print("    ├── img1001.jpg")
            print("    └── ...")
            return None, None
        
        cancer_path = os.path.join(test_path, cancer_folder)
        healthy_path = os.path.join(test_path, healthy_folder)
        
        # Contar imagens
        cancer_images = len([f for f in os.listdir(cancer_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        healthy_images = len([f for f in os.listdir(healthy_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        
        print(f"\n📊 ESTRUTURA ENCONTRADA:")
        print(f"   Pasta Câncer: {cancer_folder} ({cancer_images} imagens)")
        print(f"   Pasta Saudável: {healthy_folder} ({healthy_images} imagens)")
        print(f"   Total: {cancer_images + healthy_images} imagens")
        
        return cancer_path, healthy_path
    
    def load_labeled_test_data(self, cancer_path, healthy_path):
        """Carrega dataset de teste com labels conhecidas"""
        print("\n📁 CARREGANDO DATASET DE TESTE COM LABELS...")
        
        # Carregar câncer
        print(f"\n   Carregando imagens de câncer: {os.path.basename(cancer_path)}")
        cancer_features, cancer_files = self._load_folder_images(cancer_path)
        
        # Carregar saudáveis
        print(f"\n   Carregando imagens saudáveis: {os.path.basename(healthy_path)}")
        healthy_features, healthy_files = self._load_folder_images(healthy_path)
        
        # Combinar
        X_test = np.array(cancer_features + healthy_features)
        y_test_true = np.array([1] * len(cancer_features) + [0] * len(healthy_features))
        all_files = cancer_files + healthy_files
        
        print(f"\n📊 DATASET DE TESTE LABELED:")
        print(f"   Câncer: {len(cancer_features)} imagens")
        print(f"   Saudável: {len(healthy_features)} imagens")
        print(f"   Total: {len(X_test)} imagens")
        print(f"   Proporção: {len(cancer_features)}:{len(healthy_features)} ≈ {len(cancer_features)/len(healthy_features):.2f}:1")
        
        return X_test, y_test_true, all_files
    
    def _load_folder_images(self, folder_path, resize_dim=(128, 128), max_workers=4):
        """Carrega todas as imagens de uma pasta"""
        # Encontrar todas as imagens
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '.tiff', '.bmp', '.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
        
        image_files = list(set(image_files))
        
        features_list = []
        valid_files = []
        
        # Função para processar uma imagem
        def process_image(filepath):
            try:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None, filepath
                
                img = cv2.resize(img, resize_dim)
                img = exposure.equalize_adapthist(img)
                features = hog(img, **self.hog_parameters)
                
                return features, filepath
            except Exception:
                return None, filepath
        
        # Processamento paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_image, f): f for f in image_files}
            
            for future in tqdm(futures, total=len(image_files), desc="   Processando"):
                features, filepath = future.result()
                if features is not None:
                    features_list.append(features)
                    valid_files.append(filepath)
        
        print(f"   ✅ {len(features_list)} imagens processadas com sucesso")
        return features_list, valid_files
    
    def evaluate_on_test_set(self, X_test, y_test_true, all_files):
        """Avalia o modelo no conjunto de teste com labels conhecidas"""
        print("\n" + "=" * 80)
        print("📊 AVALIAÇÃO NO DATASET DE TESTE (LABELS CONHECIDAS)")
        print("=" * 80)
        
        # Normalizar
        print("\n⚙️  Normalizando dados de teste...")
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fazer predições
        print("🎯 Fazendo predições...")
        y_test_pred = self.svm.predict(X_test_scaled)
        y_test_proba = self.svm.predict_proba(X_test_scaled)[:, 1]
        
        # Calcular todas as métricas
        accuracy = accuracy_score(y_test_true, y_test_pred)
        cm = confusion_matrix(y_test_true, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Métricas detalhadas
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score_val = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # ROC e AUC
        fpr, tpr, thresholds = roc_curve(y_test_true, y_test_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_true, y_test_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        print(f"\n✅ RESULTADOS NO DATASET DE TESTE:")
        print(f"   Acurácia: {accuracy:.4f}")
        print(f"   Sensibilidade (Recall): {sensitivity:.4f}")
        print(f"   Especificidade: {specificity:.4f}")
        print(f"   Precisão: {precision:.4f}")
        print(f"   F1-Score: {f1_score_val:.4f}")
        print(f"   AUC-ROC: {roc_auc:.4f}")
        print(f"   AUC-PR: {pr_auc:.4f}")
        
        print(f"\n📊 MATRIZ DE CONFUSÃO:")
        print(f"              Predito")
        print(f"              Câncer  Saudável")
        print(f"Verdadeiro  Câncer   {tp:>7}  {fn:>9}")
        print(f"           Saudável  {fp:>7}  {tn:>9}")
        
        print(f"\n📈 ERROS:")
        print(f"   Falsos Positivos (saudável → câncer): {fp} ({fp/len(y_test_true)*100:.2f}%)")
        print(f"   Falsos Negativos (câncer → saudável): {fn} ({fn/len(y_test_true)*100:.2f}%)")
        
        # Relatório completo
        print(f"\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(y_test_true, y_test_pred, 
                                   target_names=['Saudável', 'Câncer'], digits=4))
        
        # Salvar resultados ANTES de visualizar
        test_results = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score_val,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(y_test_true),
            'cancer_samples': int(np.sum(y_test_true == 1)),
            'healthy_samples': int(np.sum(y_test_true == 0)),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'true_negatives': int(tn)
        }
        
        self.results['test_evaluation'] = test_results
        
        # Análise de confiança nos erros
        self._analyze_confidence_errors(y_test_true, y_test_pred, y_test_proba, all_files)
        
        # Visualizar
        self._visualize_test_results(y_test_true, y_test_pred, y_test_proba, cm)
        
        return test_results
    
    def _analyze_confidence_errors(self, y_true, y_pred, y_proba, filepaths):
        """Analisa confiança nos erros de predição"""
        print("\n🔍 ANÁLISE DE CONFIANÇA NOS ERROS:")
        
        # Identificar erros
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            print("   ✅ Nenhum erro encontrado!")
            return
        
        false_positives = []
        false_negatives = []
        
        for idx in error_indices:
            if y_true[idx] == 0 and y_pred[idx] == 1:  # FP
                false_positives.append(idx)
            elif y_true[idx] == 1 and y_pred[idx] == 0:  # FN
                false_negatives.append(idx)
        
        print(f"\n   📊 DISTRIBUIÇÃO DOS ERROS:")
        print(f"      Falsos Positivos: {len(false_positives)}")
        print(f"      Falsos Negativos: {len(false_negatives)}")
        
        # Confiança nos erros
        if false_positives:
            fp_confidences = y_proba[false_positives]  # Probabilidade de câncer
            print(f"\n   🎯 FALSOS POSITIVOS (confiança como câncer):")
            print(f"      Média: {np.mean(fp_confidences):.4f}")
            print(f"      Mínima: {np.min(fp_confidences):.4f}")
            print(f"      Máxima: {np.max(fp_confidences):.4f}")
            print(f"      < 0.7: {np.sum(fp_confidences < 0.7)}")
            print(f"      0.7-0.9: {np.sum((fp_confidences >= 0.7) & (fp_confidences < 0.9))}")
            print(f"      ≥ 0.9: {np.sum(fp_confidences >= 0.9)}")
        
        if false_negatives:
            fn_confidences = 1 - y_proba[false_negatives]  # Probabilidade de saudável
            print(f"\n   🎯 FALSOS NEGATIVOS (confiança como saudável):")
            print(f"      Média: {np.mean(fn_confidences):.4f}")
            print(f"      Mínima: {np.min(fn_confidences):.4f}")
            print(f"      Máxima: {np.max(fn_confidences):.4f}")
            print(f"      < 0.7: {np.sum(fn_confidences < 0.7)}")
            print(f"      0.7-0.9: {np.sum((fn_confidences >= 0.7) & (fn_confidences < 0.9))}")
            print(f"      ≥ 0.9: {np.sum(fn_confidences >= 0.9)}")
        
        # Salvar arquivos com erro
        self._save_error_analysis(y_true, y_pred, y_proba, filepaths, error_indices)
    
    def _save_error_analysis(self, y_true, y_pred, y_proba, filepaths, error_indices):
        """Salva análise detalhada dos erros"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = f"error_analysis_{timestamp}.csv"
        
        data = []
        for idx in error_indices:
            filename = os.path.basename(filepaths[idx])
            folder = os.path.basename(os.path.dirname(filepaths[idx]))
            
            true_label = "CÂNCER" if y_true[idx] == 1 else "SAUDÁVEL"
            pred_label = "CÂNCER" if y_pred[idx] == 1 else "SAUDÁVEL"
            error_type = "FP" if y_true[idx] == 0 else "FN"
            
            prob_cancer = y_proba[idx]
            prob_healthy = 1 - prob_cancer
            confidence = prob_cancer if pred_label == "CÂNCER" else prob_healthy
            
            data.append({
                'Arquivo': filename,
                'Pasta': folder,
                'Verdadeiro': true_label,
                'Predito': pred_label,
                'Tipo_Erro': error_type,
                'Confiança': f"{confidence:.4f}",
                'Prob_Cancer': f"{prob_cancer:.4f}",
                'Prob_Healthy': f"{prob_healthy:.4f}",
                'Caminho': filepaths[idx]
            })
        
        df = pd.DataFrame(data)
        df.to_csv(error_file, index=False, encoding='utf-8')
        print(f"\n💾 Análise de erros salva em: {error_file}")
        return error_file
    
    def _visualize_test_results(self, y_true, y_pred, y_proba, cm):
        """Visualiza resultados do teste"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Matriz de Confusão
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Saudável', 'Câncer'],
                   yticklabels=['Saudável', 'Câncer'])
        axes[0, 0].set_title('Matriz de Confusão - Dataset de Teste')
        axes[0, 0].set_ylabel('Verdadeiro')
        axes[0, 0].set_xlabel('Predito')
        
        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Curva ROC')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        axes[0, 2].plot(recall, precision, color='green', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.4f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Curva Precision-Recall')
        axes[0, 2].legend(loc="lower left")
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Distribuição de Probabilidades
        axes[1, 0].hist(y_proba[y_true == 0], bins=30, alpha=0.7, 
                       label='Saudável (Verdadeiro)', color='blue', density=True)
        axes[1, 0].hist(y_proba[y_true == 1], bins=30, alpha=0.7, 
                       label='Câncer (Verdadeiro)', color='red', density=True)
        axes[1, 0].axvline(0.5, color='black', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Probabilidade de Câncer')
        axes[1, 0].set_ylabel('Densidade')
        axes[1, 0].set_title('Distribuição por Classe Verdadeira')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Métricas por Classe
        metrics = ['Precisão', 'Recall', 'F1-Score']
        
        # Calcular métricas
        precision_scores = precision_score(y_true, y_pred, average=None)
        recall_scores = recall_score(y_true, y_pred, average=None)
        f1_scores = f1_score(y_true, y_pred, average=None)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, [precision_scores[0], recall_scores[0], f1_scores[0]], 
                      width, label='Saudável', color='blue')
        axes[1, 1].bar(x + width/2, [precision_scores[1], recall_scores[1], f1_scores[1]], 
                      width, label='Câncer', color='red')
        
        axes[1, 1].set_xlabel('Métrica')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Métricas por Classe')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Comparação Treino vs Teste
        if 'training' in self.results and 'test_evaluation' in self.results:
            train_acc = self.results['training'].get('accuracy', 0)
            test_acc = self.results['test_evaluation'].get('accuracy', 0)
            
            comparison_labels = ['Treinamento', 'Teste']
            comparison_values = [train_acc, test_acc]
            
            colors = ['lightblue', 'lightcoral']
            axes[1, 2].bar(comparison_labels, comparison_values, color=colors)
            axes[1, 2].set_title('Comparação: Acurácia Treino vs Teste')
            axes[1, 2].set_ylabel('Acurácia')
            axes[1, 2].set_ylim([0, 1.1])
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for i, v in enumerate(comparison_values):
                axes[1, 2].text(i, v + 0.01, f'{v:.4f}', 
                              ha='center', va='bottom', fontweight='bold')
            
            # Calcular diferença
            diff = train_acc - test_acc
            diff_percent = diff * 100
            if abs(diff) > 0.05:
                color = 'red' if diff > 0.05 else 'green'
                axes[1, 2].text(0.5, 0.8, f'Diferença: {diff:+.4f}\n({diff_percent:+.2f}%)', 
                              ha='center', va='center', 
                              transform=axes[1, 2].transAxes,
                              fontsize=12, color=color, fontweight='bold')
            else:
                axes[1, 2].text(0.5, 0.8, f'Diferença: {diff:+.4f}\n({diff_percent:+.2f}%)', 
                              ha='center', va='center', 
                              transform=axes[1, 2].transAxes,
                              fontsize=12, color='black')
        
        plt.suptitle('ANÁLISE COMPLETA - HOG+SVM - Dataset de Teste', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def run_complete_validation(self, cancer_train_path, healthy_train_path, test_total_path):
        """Executa validação completa com dataset de teste estruturado"""
        print("🚀 INICIANDO VALIDAÇÃO COMPLETA COM TESTE ESTRUTURADO")
        print("=" * 80)
        
        try:
            # 1. Verificar estrutura do teste
            cancer_test_path, healthy_test_path = self.check_test_structure(test_total_path)
            
            if cancer_test_path is None or healthy_test_path is None:
                print("\n❌ Por favor, organize o dataset de teste e execute novamente.")
                return None
            
            # 2. Carregar dados de treino
            print("\n" + "=" * 80)
            print("1️⃣  FASE 1: TREINAMENTO DO MODELO")
            print("=" * 80)
            
            # Carregar treino
            cancer_features, cancer_files = self._load_folder_images(cancer_train_path)
            healthy_features, healthy_files = self._load_folder_images(healthy_train_path)
            
            X_train = np.array(cancer_features + healthy_features)
            y_train = np.array([1] * len(cancer_features) + [0] * len(healthy_features))
            
            print(f"\n📊 DATASET DE TREINO:")
            print(f"   Câncer: {len(cancer_features)} imagens")
            print(f"   Saudável: {len(healthy_features)} imagens")
            print(f"   Total: {len(X_train)} imagens")
            print(f"   Proporção: {len(cancer_features)}:{len(healthy_features)} ≈ {len(cancer_features)/len(healthy_features):.2f}:1")
            
            # Split para validação interna
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Normalizar e treinar
            X_train_scaled = self.scaler.fit_transform(X_train_split)
            X_val_scaled = self.scaler.transform(X_val)
            
            print("\n🤖 Treinando modelo...")
            self.svm.fit(X_train_scaled, y_train_split)
            
            # Validar
            y_pred_val = self.svm.predict(X_val_scaled)
            accuracy_val = accuracy_score(y_val, y_pred_val)
            
            # ROC na validação
            y_proba_val = self.svm.predict_proba(X_val_scaled)[:, 1]
            fpr_val, tpr_val, _ = roc_curve(y_val, y_proba_val)
            roc_auc_val = auc(fpr_val, tpr_val)
            
            self.results['training'] = {
                'accuracy': accuracy_val,
                'roc_auc': roc_auc_val,
                'validation_size': len(X_val),
                'y_val': y_val.tolist(),
                'y_pred_val': y_pred_val.tolist(),
                'y_proba_val': y_proba_val.tolist()
            }
            
            print(f"\n✅ PERFORMANCE NA VALIDAÇÃO INTERNA:")
            print(f"   Acurácia: {accuracy_val:.4f}")
            print(f"   AUC-ROC: {roc_auc_val:.4f}")
            print(classification_report(y_val, y_pred_val, target_names=['Saudável', 'Câncer'], digits=4))
            
            # 3. Avaliar no dataset de teste estruturado
            print("\n\n" + "=" * 80)
            print("2️⃣  FASE 2: AVALIAÇÃO NO DATASET DE TESTE ESTRUTURADO")
            print("=" * 80)
            
            X_test, y_test_true, test_files = self.load_labeled_test_data(cancer_test_path, healthy_test_path)
            test_results = self.evaluate_on_test_set(X_test, y_test_true, test_files)
            
            # 4. Relatório final
            print("\n\n" + "=" * 80)
            print("3️⃣  FASE 3: RELATÓRIO FINAL E COMPARAÇÃO")
            print("=" * 80)
            
            self._generate_comparison_report(accuracy_val, test_results['accuracy'])
            
            # 5. Salvar tudo
            self._save_complete_results()
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ ERRO: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _generate_comparison_report(self, train_acc, test_acc):
        """Gera relatório de comparação entre treino e teste"""
        print(f"\n📊 COMPARAÇÃO TREINO VS TESTE:")
        print(f"   Acurácia no treino (validação): {train_acc:.4f}")
        print(f"   Acurácia no teste: {test_acc:.4f}")
        
        diff = train_acc - test_acc
        diff_percent = diff * 100
        
        print(f"   Diferença: {diff:+.4f} ({diff_percent:+.2f}%)")
        
        if abs(diff) < 0.02:
            print(f"   ✅ EXCELENTE GENERALIZAÇÃO (diferença < 2%)")
            print(f"   O modelo generaliza perfeitamente para novos dados")
        elif abs(diff) < 0.05:
            print(f"   ⚠️  BOA GENERALIZAÇÃO (diferença < 5%)")
            print(f"   O modelo generaliza bem para novos dados")
        elif abs(diff) < 0.10:
            print(f"   ⚠️  GENERALIZAÇÃO MODERADA (diferença < 10%)")
            print(f"   O modelo tem alguma perda na generalização")
        else:
            print(f"   ❌ POSSÍVEL OVERFITTING (diferença ≥ 10%)")
            print(f"   O modelo pode estar superajustado aos dados de treino")
    
    def _save_complete_results(self):
        """Salva todos os resultados"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar modelo
        model_file = f"hog_svm_complete_model_{timestamp}.pkl"
        joblib.dump({
            'svm': self.svm,
            'scaler': self.scaler,
            'hog_parameters': self.hog_parameters,
            'results': self.results
        }, model_file)
        
        # Salvar relatório detalhado
        report_file = f"complete_validation_report_{timestamp}.json"
        
        # Criar relatório completo
        full_report = {
            'timestamp': timestamp,
            'model_info': {
                'type': 'HOG + SVM Linear',
                'hog_parameters': self.hog_parameters,
                'svm_parameters': {
                    'kernel': 'linear',
                    'class_weight': 'balanced',
                    'random_state': 42
                }
            },
            'training_results': self.results.get('training', {}),
            'test_results': self.results.get('test_evaluation', {}),
            'performance_summary': {
                'train_accuracy': self.results.get('training', {}).get('accuracy', 0),
                'test_accuracy': self.results.get('test_evaluation', {}).get('accuracy', 0),
                'difference': self.results.get('training', {}).get('accuracy', 0) - 
                            self.results.get('test_evaluation', {}).get('accuracy', 0),
                'test_roc_auc': self.results.get('test_evaluation', {}).get('roc_auc', 0),
                'test_sensitivity': self.results.get('test_evaluation', {}).get('sensitivity', 0),
                'test_specificity': self.results.get('test_evaluation', {}).get('specificity', 0),
                'test_precision': self.results.get('test_evaluation', {}).get('precision', 0),
                'test_f1_score': self.results.get('test_evaluation', {}).get('f1_score', 0)
            },
            'dataset_info': {
                'training': {
                    'cancer_samples': 2918,
                    'healthy_samples': 1693,
                    'total': 4611
                },
                'test': {
                    'cancer_samples': self.results.get('test_evaluation', {}).get('cancer_samples', 0),
                    'healthy_samples': self.results.get('test_evaluation', {}).get('healthy_samples', 0),
                    'total': self.results.get('test_evaluation', {}).get('total_samples', 0)
                }
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=4, ensure_ascii=False)
        
        # Salvar resumo em texto
        summary_file = f"results_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMO DOS RESULTADOS - HOG+SVM PARA DETECÇÃO DE CÂNCER CEREBRAL\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 PERFORMANCE NO DATASET DE TESTE (NUNCA VISTO):\n")
            f.write(f"   Acurácia: {full_report['test_results'].get('accuracy', 0):.4f}\n")
            f.write(f"   AUC-ROC: {full_report['test_results'].get('roc_auc', 0):.4f}\n")
            f.write(f"   Sensibilidade: {full_report['test_results'].get('sensitivity', 0):.4f}\n")
            f.write(f"   Especificidade: {full_report['test_results'].get('specificity', 0):.4f}\n")
            f.write(f"   Precisão: {full_report['test_results'].get('precision', 0):.4f}\n")
            f.write(f"   F1-Score: {full_report['test_results'].get('f1_score', 0):.4f}\n\n")
            
            f.write("📊 MATRIZ DE CONFUSÃO (TESTE):\n")
            cm = full_report['test_results'].get('confusion_matrix', [[0, 0], [0, 0]])
            f.write(f"   Verdadeiro Câncer → Predito Câncer: {cm[1][1]}\n")
            f.write(f"   Verdadeiro Câncer → Predito Saudável: {cm[1][0]}\n")
            f.write(f"   Verdadeiro Saudável → Predito Câncer: {cm[0][1]}\n")
            f.write(f"   Verdadeiro Saudável → Predito Saudável: {cm[0][0]}\n\n")
            
            f.write("📈 COMPARAÇÃO TREINO-TESTE:\n")
            f.write(f"   Treino (validação): {full_report['training_results'].get('accuracy', 0):.4f}\n")
            f.write(f"   Teste: {full_report['test_results'].get('accuracy', 0):.4f}\n")
            f.write(f"   Diferença: {full_report['performance_summary']['difference']:+.4f}\n\n")
            
            f.write("🎯 CONCLUSÃO:\n")
            diff = full_report['performance_summary']['difference']
            if abs(diff) < 0.02:
                f.write("   ✅ Excelente generalização - modelo robusto e confiável\n")
            elif abs(diff) < 0.05:
                f.write("   ⚠️  Boa generalização - modelo confiável\n")
            else:
                f.write("   ❌ Possível overfitting - requer atenção\n")
            
            f.write(f"\n📅 Data da análise: {timestamp}\n")
            f.write("=" * 80)
        
        print(f"\n💾 ARQUIVOS SALVOS:")
        print(f"   1. Modelo completo: {model_file}")
        print(f"   2. Relatório detalhado (JSON): {report_file}")
        print(f"   3. Resumo dos resultados (TXT): {summary_file}")
        
        if 'test_evaluation' in self.results:
            test_acc = self.results['test_evaluation'].get('accuracy', 0)
            print(f"\n✅ RESULTADO FINAL: {test_acc:.4f} de acurácia no teste independente")
        
        print("\n" + "=" * 80)
        print("🎉 VALIDAÇÃO COMPLETA CONCLUÍDA!")
        print("=" * 80)

# ==================== EXECUÇÃO PRINCIPAL ====================

if __name__ == "__main__":
    # CONFIGURAÇÕES
    CANCER_TRAIN = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Cancer_brain"
    HEALTHY_TRAIN = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Healthy_brain"
    TEST_TOTAL = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Test_Images_Total"
    
    print("=" * 80)
    print("🧪 VALIDAÇÃO COMPLETA HOG+SVM COM DATASET DE TESTE ESTRUTURADO")
    print("=" * 80)
    print(f"📁 Treino - Câncer: {CANCER_TRAIN}")
    print(f"📁 Treino - Saudável: {HEALTHY_TRAIN}")
    print(f"📁 Teste Estruturado: {TEST_TOTAL}")
    print("=" * 80)
    print("\n⚠️  PRÉ-REQUISITO:")
    print("O dataset de teste DEVE estar organizado assim:")
    print("Test_Images_Total/")
    print("├── Cancer/           # Todas imagens com câncer")
    print("│   ├── img1.jpg")
    print("│   └── ...")
    print("└── Healthy/          # Todas imagens saudáveis")
    print("    ├── img1001.jpg")
    print("    └── ...")
    print("=" * 80)
    
    # Executar validação
    validator = HOG_SVM_CompleteValidator()
    results = validator.run_complete_validation(CANCER_TRAIN, HEALTHY_TRAIN, TEST_TOTAL)
    
    if results:
        print("\n✅ PRONTO PARA PUBLICAÇÃO CIENTÍFICA!")
        print("Você agora tem:")
        print("1. 📊 Matriz de confusão completa")
        print("2. 📈 Gráficos de análise (ROC, distribuição, etc.)")
        print("3. 🧠 Modelo treinado salvo (.pkl)")
        print("4. 📄 Relatório completo (JSON e TXT)")
        print("5. 🔍 Análise detalhada dos erros (.csv)")
        print("\n📊 RESULTADOS OBTIDOS:")
        print(f"   Acurácia no treino: {results.get('training', {}).get('accuracy', 0):.4f}")
        print(f"   Acurácia no teste: {results.get('test_evaluation', {}).get('accuracy', 0):.4f}")
        print(f"   AUC-ROC no teste: {results.get('test_evaluation', {}).get('roc_auc', 0):.4f}")
        print(f"   Sensibilidade: {results.get('test_evaluation', {}).get('sensitivity', 0):.4f}")
        print(f"   Especificidade: {results.get('test_evaluation', {}).get('specificity', 0):.4f}")
        print("=" * 80)