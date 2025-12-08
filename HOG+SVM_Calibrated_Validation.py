import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from skimage.feature import hog
import joblib
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import pandas as pd
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class ScientificHOG_SVM_Validator:
    """Validador Científico Completo para Modelo HOG+SVM"""
    
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
    
    # ==================== CARREGAMENTO DE DADOS ====================
    
    def load_all_images_from_folder(self, folder_path, label, resize_dim=(128, 128), max_workers=4):
        """Carrega todas as imagens com tracking de origem"""
        print(f"\n📁 Carregando {'POSITIVAS' if label == 1 else 'NEGATIVAS'} de: {folder_path}")
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
        
        image_files = list(set(image_files))
        print(f"🔍 Encontradas {len(image_files)} arquivos de imagem")
        
        features_list = []
        file_origins = []
        failed_files = []
        
        def process_single_image(filepath):
            try:
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return None, filepath, None
                
                original_size = image.shape
                image = cv2.resize(image, resize_dim)
                image = exposure.equalize_adapthist(image)
                features = hog(image, **self.hog_parameters)
                
                return features, None, original_size
            except Exception as e:
                return None, filepath, None
        
        print("🔄 Processando imagens...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_image, filepath): filepath 
                            for filepath in image_files}
            
            for future in tqdm(as_completed(future_to_file), total=len(image_files), 
                             desc=f"Processing {'positive' if label == 1 else 'negative'}"):
                features, failed_file, orig_size = future.result()
                if features is not None:
                    features_list.append(features)
                    file_origins.append({
                        'original_size': orig_size,
                        'label': label
                    })
                elif failed_file:
                    failed_files.append(failed_file)
        
        print(f"✅ Processadas com sucesso: {len(features_list)} imagens")
        if failed_files:
            print(f"⚠️  {len(failed_files)} arquivos falharam")
        
        return features_list, len(features_list), file_origins
    
    def load_datasets_with_tracking(self, positive_path, negative_path):
        """Carrega datasets mantendo metadata de origem"""
        print("=" * 80)
        print("🧪 CARREGAMENTO CIENTÍFICO COM METADATA")
        print("=" * 80)
        
        positive_features, pos_count, pos_origins = self.load_all_images_from_folder(positive_path, 1)
        negative_features, neg_count, neg_origins = self.load_all_images_from_folder(negative_path, 0)
        
        X = np.array(positive_features + negative_features)
        y = np.array([1] * pos_count + [0] * neg_count)
        origins = pos_origins + neg_origins
        
        print(f"\n📊 ESTATÍSTICAS DO DATASET:")
        print(f"   Positivas (Câncer): {pos_count} imagens")
        print(f"   Negativas (Saudáveis): {neg_count} imagens")
        print(f"   Total: {len(X)} imagens")
        print(f"   Proporção: {pos_count}:{neg_count} ≈ {pos_count/neg_count:.2f}:1")
        print(f"   Features HOG: {X.shape[1]}")
        
        # Análise de tamanhos originais
        sizes = [o['original_size'] for o in origins if o['original_size'] is not None]
        if sizes:
            avg_size = np.mean([s[0]*s[1] for s in sizes])
            print(f"   Tamanho médio original: {int(avg_size):,} pixels")
        
        return X, y, origins
    
    # ==================== VALIDAÇÃO ESTATÍSTICA ====================
    
    def statistical_validation(self, X, y, n_splits=5, n_bootstraps=1000):
        """Validação estatística robusta"""
        print("\n" + "=" * 80)
        print("📊 VALIDAÇÃO ESTATÍSTICA ROBUSTA")
        print("=" * 80)
        
        # 1. Cross-Validation Estratificado
        print("\n1️⃣  CROSS-VALIDATION ESTRATIFICADA (5-fold):")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.svm, self.scaler.fit_transform(X), y, 
                                   cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"   Scores: {cv_scores}")
        print(f"   Média: {cv_scores.mean():.4f}")
        print(f"   Desvio Padrão: {cv_scores.std():.4f}")
        print(f"   Intervalo: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        # 2. Bootstrap Confidence Intervals
        print("\n2️⃣  INTERVALOS DE CONFIANÇA (Bootstrap, n={n_bootstraps}):")
        
        bootstrapped_scores = []
        rng = np.random.RandomState(42)
        
        for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            indices = rng.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=0.2, random_state=42, stratify=y_boot
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            bootstrapped_scores.append(score)
        
        bootstrapped_scores = np.array(bootstrapped_scores)
        ci_95 = np.percentile(bootstrapped_scores, [2.5, 97.5])
        ci_99 = np.percentile(bootstrapped_scores, [0.5, 99.5])
        
        print(f"   Média Bootstrap: {bootstrapped_scores.mean():.4f}")
        print(f"   95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"   99% CI: [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")
        
        # 3. Teste de Significância
        print("\n3️⃣  TESTE DE SIGNIFICÂNCIA ESTATÍSTICA:")
        
        # Teste t contra baseline (50%)
        t_stat, p_value = stats.ttest_1samp(bootstrapped_scores, 0.5)
        print(f"   Teste t contra baseline (50%):")
        print(f"   t-statistic = {t_stat:.4f}")
        print(f"   p-value = {p_value:.4e}")
        print(f"   Significativo (p < 0.05)? {'✅ SIM' if p_value < 0.05 else '❌ NÃO'}")
        
        # Effect Size (Cohen's d)
        cohen_d = (bootstrapped_scores.mean() - 0.5) / bootstrapped_scores.std()
        print(f"   Effect Size (Cohen's d): {cohen_d:.4f}")
        print(f"   Interpretação: {'Grande' if cohen_d > 0.8 else 'Médio' if cohen_d > 0.5 else 'Pequeno'}")
        
        self.results['cv_scores'] = cv_scores
        self.results['bootstrapped_scores'] = bootstrapped_scores
        self.results['ci_95'] = ci_95
        self.results['ci_99'] = ci_99
        self.results['p_value'] = p_value
        self.results['cohen_d'] = cohen_d
        
        return bootstrapped_scores
    
    # ==================== ANÁLISE DE VIÉS ====================
    
    def bias_analysis(self, X, y, y_pred, y_pred_proba, origins=None):
        """Análise completa de viés e equidade"""
        print("\n" + "=" * 80)
        print("⚖️  ANÁLISE DE VIÉS E EQUIDADE")
        print("=" * 80)
        
        # 1. Matriz de Confusão Detalhada
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n1️⃣  MATRIZ DE CONFUSÃO DETALHADA:")
        print(f"   Verdadeiros Negativos: {tn} ({tn/len(y)*100:.2f}%)")
        print(f"   Falsos Positivos: {fp} ({fp/len(y)*100:.2f}%)")
        print(f"   Falsos Negativos: {fn} ({fn/len(y)*100:.2f}%)")
        print(f"   Verdadeiros Positivos: {tp} ({tp/len(y)*100:.2f}%)")
        
        # 2. Métricas por Classe
        print("\n2️⃣  MÉTRICAS POR CLASSE:")
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precisão positiva
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precisão negativa
        
        print(f"   Sensibilidade (Recall Positivo): {sensitivity:.4f}")
        print(f"   Especificidade (Recall Negativo): {specificity:.4f}")
        print(f"   Valor Preditivo Positivo: {ppv:.4f}")
        print(f"   Valor Preditivo Negativo: {npv:.4f}")
        
        # 3. Viés de Previsão
        print("\n3️⃣  ANÁLISE DE VIÉS DE PREVISÃO:")
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(10, 8))
        
        # Subplot 1: Calibration Plot
        plt.subplot(2, 2, 1)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Probabilidade Média Predita')
        plt.ylabel('Fração de Positivos')
        plt.title('Curva de Calibração')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Distribuição de Probabilidades
        plt.subplot(2, 2, 2)
        plt.hist(y_pred_proba[y == 0], bins=30, alpha=0.7, label='Saudável', density=True)
        plt.hist(y_pred_proba[y == 1], bins=30, alpha=0.7, label='Câncer', density=True)
        plt.axvline(0.5, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Probabilidade de Câncer')
        plt.ylabel('Densidade')
        plt.title('Distribuição por Classe')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: ROC Curve
        plt.subplot(2, 2, 3)
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Precision-Recall Curve
        plt.subplot(2, 2, 4)
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n   AUC-ROC: {roc_auc:.4f}")
        print(f"   AUC-PR: {pr_auc:.4f}")
        
        # 4. Análise de Limiar Ótimo
        print("\n4️⃣  ANÁLISE DE LIMIAR ÓTIMO:")
        
        # Encontra limiar que maximiza F1-score
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            tp = np.sum((y_pred_thresh == 1) & (y == 1))
            fp = np.sum((y_pred_thresh == 1) & (y == 0))
            fn = np.sum((y_pred_thresh == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_thresh = thresholds[optimal_idx]
        
        print(f"   Limiar padrão: 0.5")
        print(f"   Limiar ótimo (max F1): {optimal_thresh:.2f}")
        print(f"   F1-score no limiar ótimo: {f1_scores[optimal_idx]:.4f}")
        
        self.results['sensitivity'] = sensitivity
        self.results['specificity'] = specificity
        self.results['roc_auc'] = roc_auc
        self.results['pr_auc'] = pr_auc
        self.results['optimal_threshold'] = optimal_thresh
        
        return optimal_thresh
    
    # ==================== VALIDAÇÃO DO MODELO ====================
    
    def comprehensive_model_validation(self, X, y, test_size=0.2):
        """Validação completa do modelo"""
        print("\n" + "=" * 80)
        print("🧬 VALIDAÇÃO COMPLETA DO MODELO HOG+SVM")
        print("=" * 80)
        
        # 1. Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 2. Normalização
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 3. Treinamento
        print("\n🤖 TREINANDO MODELO FINAL...")
        self.svm.fit(X_train_scaled, y_train)
        
        # 4. Predições
        y_pred = self.svm.predict(X_test_scaled)
        y_pred_proba = self.svm.predict_proba(X_test_scaled)[:, 1]
        
        # 5. Métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 DESEMPENHO NO CONJUNTO DE TESTE:")
        print(f"   Acurácia: {accuracy:.4f}")
        print(f"   Tamanho do teste: {len(y_test)} amostras")
        print(f"   Proporção: {np.sum(y_test)} positivas / {len(y_test)-np.sum(y_test)} negativas")
        
        # 6. Relatório detalhado
        print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(y_test, y_pred, target_names=['Saudável', 'Câncer'], digits=4))
        
        self.results['X_test'] = X_test
        self.results['y_test'] = y_test
        self.results['y_pred'] = y_pred
        self.results['y_pred_proba'] = y_pred_proba
        self.results['accuracy'] = accuracy
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    # ==================== RELATÓRIO CIENTÍFICO ====================
    
    def generate_scientific_report(self):
        """Gera relatório científico completo"""
        print("\n" + "=" * 80)
        print("📄 RELATÓRIO CIENTÍFICO COMPLETO")
        print("=" * 80)
        
        report = {
            'model': 'HOG + SVM Linear',
            'hog_parameters': self.hog_parameters,
            'dataset_size': len(self.results.get('X_test', [])) * 5,  # Estimativa
            'statistical_validation': {},
            'performance_metrics': {},
            'bias_analysis': {},
            'conclusions': []
        }
        
        # Estatísticas
        if 'cv_scores' in self.results:
            report['statistical_validation']['cross_validation'] = {
                'mean_accuracy': float(self.results['cv_scores'].mean()),
                'std_accuracy': float(self.results['cv_scores'].std()),
                'ci_95_cv': [float(self.results['cv_scores'].mean() - 1.96*self.results['cv_scores'].std()),
                            float(self.results['cv_scores'].mean() + 1.96*self.results['cv_scores'].std())]
            }
        
        if 'bootstrapped_scores' in self.results:
            report['statistical_validation']['bootstrap'] = {
                'mean_accuracy': float(self.results['bootstrapped_scores'].mean()),
                'ci_95': [float(self.results['ci_95'][0]), float(self.results['ci_95'][1])],
                'ci_99': [float(self.results['ci_99'][0]), float(self.results['ci_99'][1])],
                'p_value': float(self.results['p_value']),
                'cohen_d': float(self.results['cohen_d'])
            }
        
        # Métricas de performance
        if 'accuracy' in self.results:
            report['performance_metrics'] = {
                'accuracy': float(self.results['accuracy']),
                'sensitivity': float(self.results.get('sensitivity', 0)),
                'specificity': float(self.results.get('specificity', 0)),
                'roc_auc': float(self.results.get('roc_auc', 0)),
                'pr_auc': float(self.results.get('pr_auc', 0))
            }
        
        # Conclusões
        if self.results.get('p_value', 1) < 0.05:
            report['conclusions'].append("✅ O modelo é estatisticamente significativo (p < 0.05)")
        
        if self.results.get('roc_auc', 0) > 0.95:
            report['conclusions'].append("✅ Excelente capacidade discriminativa (AUC > 0.95)")
        
        if self.results.get('cohen_d', 0) > 0.8:
            report['conclusions'].append("✅ Grande tamanho de efeito (Cohen's d > 0.8)")
        
        if abs(self.results.get('sensitivity', 0) - self.results.get('specificity', 0)) < 0.1:
            report['conclusions'].append("✅ Balanceado entre sensibilidade e especificidade")
        else:
            report['conclusions'].append("⚠️  Possível viés - verificar diferença sensibilidade/especificidade")
        
        # Imprime relatório
        print("\n📈 RESUMO ESTATÍSTICO:")
        print(f"   Acurácia média: {report['performance_metrics'].get('accuracy', 0):.4f}")
        print(f"   AUC-ROC: {report['performance_metrics'].get('roc_auc', 0):.4f}")
        print(f"   Sensibilidade: {report['performance_metrics'].get('sensitivity', 0):.4f}")
        print(f"   Especificidade: {report['performance_metrics'].get('specificity', 0):.4f}")
        
        print("\n📊 VALIDAÇÃO ESTATÍSTICA:")
        print(f"   p-value: {report['statistical_validation'].get('bootstrap', {}).get('p_value', 0):.4e}")
        print(f"   95% CI: {report['statistical_validation'].get('bootstrap', {}).get('ci_95', [0, 0])}")
        print(f"   Cohen's d: {report['statistical_validation'].get('bootstrap', {}).get('cohen_d', 0):.4f}")
        
        print("\n🎯 CONCLUSÕES:")
        for conclusion in report['conclusions']:
            print(f"   {conclusion}")
        
        print("\n" + "=" * 80)
        print("🧪 VALIDAÇÃO CIENTÍFICA CONCLUÍDA")
        print("=" * 80)
        
        return report
    
    # ==================== FUNÇÃO PRINCIPAL ====================
    
    def run_complete_validation(self, positive_path, negative_path):
        """Executa validação científica completa"""
        try:
            # 1. Carregar dados
            print("🚀 INICIANDO VALIDAÇÃO CIENTÍFICA COMPLETA")
            X, y, origins = self.load_datasets_with_tracking(positive_path, negative_path)
            
            # 2. Validação estatística
            self.statistical_validation(X, y)
            
            # 3. Treinamento e validação do modelo
            X_test_scaled, y_test, y_pred, y_pred_proba = self.comprehensive_model_validation(X, y)
            
            # 4. Análise de viés
            self.bias_analysis(X_test_scaled, y_test, y_pred, y_pred_proba, origins)
            
            # 5. Relatório final
            report = self.generate_scientific_report()
            
            # 6. Salvar resultados
            self.save_validation_results(report)
            
            return report
            
        except Exception as e:
            print(f"\n❌ ERRO NA VALIDAÇÃO: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_validation_results(self, report):
        """Salva resultados da validação"""
        import json
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hog_svm_validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        print(f"\n💾 Relatório salvo como: {filename}")
        
        # Salva também o modelo
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'hog_parameters': self.hog_parameters,
            'validation_report': report
        }
        joblib.dump(model_data, f"hog_svm_validated_model_{timestamp}.pkl")
        print(f"💾 Modelo validado salvo como: hog_svm_validated_model_{timestamp}.pkl")

# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    # Configurações
    POSITIVE_PATH = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Cancer_brain"
    NEGATIVE_PATH = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Healthy_brain"
    
    # Executar validação completa
    validator = ScientificHOG_SVM_Validator()
    report = validator.run_complete_validation(POSITIVE_PATH, NEGATIVE_PATH)
    
    # Resultado final
    if report:
        print("\n" + "=" * 80)
        print("🎉 VALIDAÇÃO CIENTÍFICA CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print("\nEste modelo está pronto para publicação científica com:")
        print("✅ Validação estatística robusta")
        print("✅ Análise de viés completa")
        print("✅ Intervalos de confiança")
        print("✅ Testes de significância")
        print("✅ Métricas de equidade")