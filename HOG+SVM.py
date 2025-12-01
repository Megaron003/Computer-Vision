import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from skimage.feature import hog
import joblib

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
            
        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_hog_features(self, image_path, resize_dim=(128, 128)):
        """Extrai features HOG de uma imagem"""
        try:
            # Carrega a imagem em escala de cinza
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Erro ao carregar: {image_path}")
                return None
            
            # Redimensiona para tamanho padrão
            image = cv2.resize(image, resize_dim)
            
            # Aplica equalização de histograma para melhor contraste
            image = exposure.equalize_adapthist(image)
            
            # Extrai features HOG
            features = hog(image, **self.hog_parameters)
            
            return features
        except Exception as e:
            print(f"Erro processando {image_path}: {str(e)}")
            return None
    
    def load_dataset(self, positive_path, negative_path, test_size=0.2):
        """Carrega e processa o dataset"""
        print("Carregando imagens positivas...")
        positive_features = []
        positive_labels = []
        
        for filename in os.listdir(positive_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                filepath = os.path.join(positive_path, filename)
                features = self.extract_hog_features(filepath)
                if features is not None:
                    positive_features.append(features)
                    positive_labels.append(1)  # 1 para positivo (câncer)
        
        print("Carregando imagens negativas...")
        negative_features = []
        negative_labels = []
        
        for filename in os.listdir(negative_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                filepath = os.path.join(negative_path, filename)
                features = self.extract_hog_features(filepath)
                if features is not None:
                    negative_features.append(features)
                    negative_labels.append(0)  # 0 para negativo (saudável)
        
        # Combina os dados
        X = np.array(positive_features + negative_features)
        y = np.array(positive_labels + negative_labels)
        
        print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
        print(f"Distribuição: {np.sum(y)} positivas, {len(y) - np.sum(y)} negativas")
        
        return X, y
    
    def train(self, X, y):
        """Treina o classificador SVM"""
        print("Iniciando treinamento...")
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normaliza os dados
        print("Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treina a SVM
        print("Treinando SVM...")
        self.svm.fit(X_train_scaled, y_train)
        
        # Avalia no conjunto de teste
        y_pred = self.svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia no teste: {accuracy:.4f}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, target_names=['Saudável', 'Câncer']))
        
        self.is_trained = True
        
        # Retorna métricas para análise
        return {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy
        }
    
    def predict(self, image_path):
        """Faz predição em uma nova imagem"""
        if not self.is_trained:
            raise Exception("Modelo não treinado!")
        
        features = self.extract_hog_features(image_path)
        if features is None:
            return None
        
        features_scaled = self.scaler.transform([features])
        prediction = self.svm.predict(features_scaled)[0]
        probability = self.svm.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'Câncer' if prediction == 1 else 'Saudável',
            'confidence': probability[1] if prediction == 1 else probability[0],
            'probabilities': {
                'Saudável': probability[0],
                'Câncer': probability[1]
            }
        }
    
    def evaluate_model(self, X_test, y_test, y_pred):
        """Avaliação detalhada do modelo"""
        print("\n=== AVALIAÇÃO DETALHADA ===")
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Saudável', 'Câncer'],
                   yticklabels=['Saudável', 'Câncer'])
        plt.title('Matriz de Confusão - HOG + SVM')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.show()
        
        # Métricas detalhadas
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Saudável', 'Câncer']))
    
    def save_model(self, filename):
        """Salva o modelo treinado"""
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'hog_parameters': self.hog_parameters,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filename)
        print(f"Modelo salvo como {filename}")
    
    def load_model(self, filename):
        """Carrega um modelo salvo"""
        model_data = joblib.load(filename)
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.hog_parameters = model_data['hog_parameters']
        self.is_trained = model_data['is_trained']
        print(f"Modelo carregado de {filename}")

# USO DO CÓDIGO
if __name__ == "__main__":
    # Configurações
    POSITIVE_PATH = "C:/Users/GuilhermeBragadoVale/Downloads/axial MRI.v2-release.yolov8/train/images"  # Imagens com câncer
    NEGATIVE_PATH = "C:/Users/GuilhermeBragadoVale/Desktop/Computer_Vision_For_Health/Cerebral_Cancer/Dataset/Healthy_brain"  # Imagens saudáveis
    
    # Cria e treina o classificador
    classifier = HOG_SVM_Classifier()
    
    # Carrega o dataset
    X, y = classifier.load_dataset(POSITIVE_PATH, NEGATIVE_PATH)
    
    # Treina o modelo
    results = classifier.train(X, y)
    
    # Avaliação detalhada
    classifier.evaluate_model(results['X_test'], results['y_test'], results['y_pred'])
    
    # Salva o modelo
    classifier.save_model("hog_svm_brain_cancer.pkl")
    
    # Exemplo de predição em uma nova imagem
    # result = classifier.predict("caminho/para/nova/imagem.jpg")
    # print(f"Predição: {result['prediction']}")
    # print(f"Confiança: {result['confidence']:.4f}")