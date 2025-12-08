import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
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

class HOG_SVM_Classifier:
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
        self.is_trained = False
    
    def load_all_images_from_folder(self, folder_path, label, resize_dim=(128, 128), max_workers=4):
        """Carrega TODAS as imagens de uma pasta usando processamento paralelo"""
        print(f"\n📁 Carregando {'POSITIVAS' if label == 1 else 'NEGATIVAS'} de: {folder_path}")
        
        # Busca TODOS os arquivos de imagem
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
        
        # Remove duplicatas
        image_files = list(set(image_files))
        
        print(f"🔍 Encontradas {len(image_files)} arquivos de imagem")
        
        features_list = []
        failed_files = []
        
        # Função para processar uma única imagem
        def process_single_image(filepath):
            try:
                # Carrega imagem em escala de cinza
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return None, filepath
                
                # Redimensiona para tamanho padrão
                image = cv2.resize(image, resize_dim)
                
                # Aplica equalização de histograma
                image = exposure.equalize_adapthist(image)
                
                # Extrai features HOG
                features = hog(image, **self.hog_parameters)
                
                return features, None
                
            except Exception as e:
                return None, filepath
        
        # Processamento paralelo para maior velocidade
        print("🔄 Processando imagens...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submete todas as tarefas
            future_to_file = {executor.submit(process_single_image, filepath): filepath 
                            for filepath in image_files}
            
            # Coleta os resultados
            for future in tqdm(as_completed(future_to_file), total=len(image_files), 
                             desc=f"Processing {'positive' if label == 1 else 'negative'}"):
                features, failed_file = future.result()
                if features is not None:
                    features_list.append(features)
                elif failed_file:
                    failed_files.append(failed_file)
        
        count = len(features_list)
        
        if failed_files:
            print(f"⚠️  {len(failed_files)} arquivos falharam no processamento")
            if len(failed_files) <= 10:  # Mostra apenas os primeiros 10
                for failed in failed_files[:10]:
                    print(f"   - {os.path.basename(failed)}")
            else:
                print(f"   (Mostrando 10 de {len(failed_files)})")
                for failed in failed_files[:10]:
                    print(f"   - {os.path.basename(failed)}")
        
        print(f"✅ Processadas com sucesso: {count} imagens")
        
        return features_list, count
    
    def load_dataset_complete(self, positive_path, negative_path):
        """Carrega TODAS as imagens disponíveis"""
        print("=" * 70)
        print("📦 CARREGAMENTO COMPLETO DO DATASET")
        print("=" * 70)
        
        print("\n🔎 Buscando todas as imagens disponíveis...")
        
        # Carrega imagens positivas
        positive_features, positive_count = self.load_all_images_from_folder(positive_path, label=1)
        
        # Carrega imagens negativas  
        negative_features, negative_count = self.load_all_images_from_folder(negative_path, label=0)
        
        # Verifica se há dados suficientes
        if positive_count == 0:
            raise ValueError("❌ Nenhuma imagem positiva encontrada!")
        if negative_count == 0:
            raise ValueError("❌ Nenhuma imagem negativa encontrada!")
        
        # Combina os dados
        X = np.array(positive_features + negative_features)
        y = np.array([1] * positive_count + [0] * negative_count)
        
        print("\n" + "=" * 70)
        print("📊 ESTATÍSTICAS COMPLETAS DO DATASET")
        print("=" * 70)
        print(f"   Positivas (Câncer): {positive_count} imagens")
        print(f"   Negativas (Saudáveis): {negative_count} imagens")
        print(f"   Total: {len(X)} imagens")
        print(f"   Proporção: {positive_count}:{negative_count} ≈ {positive_count/negative_count:.2f}:1")
        print(f"   Features por imagem: {X.shape[1]}")
        print(f"   Memória utilizada: {X.nbytes / (1024**2):.2f} MB")
        print("=" * 70)
        
        return X, y
    
    def train_with_validation(self, X, y, test_size=0.2):
        """Treina com validação e análise detalhada"""
        print("\n" + "=" * 60)
        print("🎯 INICIANDO TREINAMENTO HOG + SVM")
        print("=" * 60)
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📈 DIVISÃO DOS DADOS:")
        print(f"   Conjunto completo: {X.shape[0]} amostras")
        print(f"   Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
        
        # Balanceamento no conjunto de teste
        test_positives = np.sum(y_test)
        test_negatives = len(y_test) - test_positives
        print(f"   Proporção no teste: {test_positives} positivas / {test_negatives} negativas")
        
        # Normaliza os dados
        print("\n⚙️  Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treina a SVM
        print("🤖 Treinando SVM (isso pode levar alguns minutos)...")
        self.svm.fit(X_train_scaled, y_train)
        
        # Avalia no conjunto de teste
        print("📊 Avaliando modelo...")
        y_pred = self.svm.predict(X_test_scaled)
        y_pred_proba = self.svm.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ RESULTADOS DO TREINAMENTO:")
        print(f"   Acurácia: {accuracy:.4f}")
        print(f"\n📋 RELATÓRIO DETALHADO:")
        print(classification_report(y_test, y_pred, target_names=['Saudável', 'Câncer'], digits=4))
        
        self.is_trained = True
        
        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'model': self.svm,
            'dataset_size': X.shape[0]
        }
    
    def comprehensive_evaluation(self, results):
        """Avaliação abrangente com visualizações"""
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        dataset_size = results.get('dataset_size', 0)
        
        print("\n" + "=" * 70)
        print("📈 AVALIAÇÃO COMPREENSIVA DO MODELO")
        print("=" * 70)
        
        # 1. Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Matriz de Confusão
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Saudável', 'Câncer'],
                   yticklabels=['Saudável', 'Câncer'])
        axes[0, 0].set_title(f'Matriz de Confusão (Teste: {len(y_test)} amostras)')
        axes[0, 0].set_ylabel('Verdadeiro')
        axes[0, 0].set_xlabel('Predito')
        
        # 2. Curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Curva ROC')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribuição das Probabilidades
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                       label='Saudável', color='blue', density=True)
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                       label='Câncer', color='red', density=True)
        axes[1, 0].set_xlabel('Probabilidade de ser Câncer')
        axes[1, 0].set_ylabel('Densidade')
        axes[1, 0].set_title('Distribuição das Probabilidades')
        axes[1, 0].legend()
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Métricas por Classe
        report = classification_report(y_test, y_pred, 
                                     target_names=['Saudável', 'Câncer'],
                                     output_dict=True)
        
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Saudável', 'Câncer']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, cls in enumerate(classes):
            values = [report[cls][metric] for metric in metrics]
            axes[1, 1].bar(x + i*width, values, width, label=cls, 
                          color='blue' if cls == 'Saudável' else 'red')
        
        axes[1, 1].set_xlabel('Métricas')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Métricas por Classe')
        axes[1, 1].set_xticks(x + width/2)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'HOG+SVM - Detecção de Câncer Cerebral (Dataset: {dataset_size} imagens)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Métricas numéricas detalhadas
        print(f"\n📊 MÉTRICAS DETALHADAS:")
        print(f"   Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        print(f"   AUC-ROC: {roc_auc:.4f}")
        print(f"   Precisão (Câncer): {report['Câncer']['precision']:.4f}")
        print(f"   Recall (Câncer): {report['Câncer']['recall']:.4f}")
        print(f"   F1-Score (Câncer): {report['Câncer']['f1-score']:.4f}")
        
        # Importância das features (coeficientes da SVM)
        if hasattr(self.svm, 'coef_'):
            feature_importance = np.abs(self.svm.coef_[0])
            print(f"\n🔍 ANÁLISE DAS FEATURES HOG:")
            print(f"   Total de features HOG: {len(feature_importance)}")
            print(f"   Feature mais importante: {np.max(feature_importance):.4f}")
            print(f"   Feature menos importante: {np.min(feature_importance):.4f}")
            print(f"   Importância média: {np.mean(feature_importance):.4f}")
            print(f"   Desvio padrão: {np.std(feature_importance):.4f}")
        
        return roc_auc
    
    def save_model(self, filename):
        """Salva o modelo treinado"""
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'hog_parameters': self.hog_parameters,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filename)
        print(f"💾 Modelo salvo como: {filename}")
    
    def load_model(self, filename):
        """Carrega um modelo salvo"""
        model_data = joblib.load(filename)
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.hog_parameters = model_data['hog_parameters']
        self.is_trained = model_data['is_trained']
        print(f"📂 Modelo carregado de: {filename}")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    # Configurações dos paths
    POSITIVE_PATH = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Cancer_brain"
    NEGATIVE_PATH = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Healthy_brain"
    
    try:
        # Cria o classificador
        print("🚀 Inicializando classificador HOG+SVM...")
        classifier = HOG_SVM_Classifier()
        
        # Carrega TODAS as imagens disponíveis
        print("\n" + "=" * 70)
        print("📂 INICIANDO CARREGAMENTO COMPLETO DO DATASET")
        print("=" * 70)
        X, y = classifier.load_dataset_complete(POSITIVE_PATH, NEGATIVE_PATH)
        
        # Treina o modelo
        print("\n🎯 INICIANDO PROCESSO DE TREINAMENTO...")
        results = classifier.train_with_validation(X, y, test_size=0.2)
        
        # Avaliação completa
        print("\n📈 GERANDO ANÁLISE COMPLETA DOS RESULTADOS...")
        roc_auc = classifier.comprehensive_evaluation(results)
        
        # Salva o modelo
        classifier.save_model("hog_svm_brain_cancer_complete.pkl")
        
        print("\n" + "=" * 70)
        print("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        print(f"📊 Resultado Final:")
        print(f"   Dataset: {results['dataset_size']} imagens")
        print(f"   Acurácia: {results['accuracy']:.4f}")
        print(f"   AUC-ROC: {roc_auc:.4f}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {str(e)}")
        print("Verifique os caminhos das pastas e as permissões de acesso.")