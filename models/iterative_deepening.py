import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
import time
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class IDDFSVacationRecommender:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.destinations = []
        self.destination_encodings = {}
        self.destination_profiles = {}
        self.max_depth = 10  # Maksimum arama derinliği
        
    def train(self, df):
        """IDDFS tabanlı tatil önerici modeli eğit"""
        logger.info("IDDFS tabanlı model eğitimi başlıyor...")
        
        # Gereksiz sütunları kaldır
        columns_to_drop = ['user_id', 'hotel_price_per_night', 'flight_cost', 'total_cost']
        df = df.drop(columns_to_drop, axis=1)
        
        # Kategorik değişkenleri encode et
        categorical_features = ['season', 'preferred_activity', 'destination']
        
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
                # Destinasyonları sakla
                if col == 'destination':
                    self.destinations = list(le.classes_)
                    # Her destinasyon için encoding değerini sakla
                    self.destination_encodings = {le.classes_[i]: i for i in range(len(le.classes_))}
        
        # Sayısal değişkenleri ölçeklendir
        numerical_features = ['budget', 'duration', 'value_score', 'user_satisfaction']
        self.scaler = StandardScaler()
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Feature'ları ve target'ı ayır
        self.feature_columns = ['season', 'preferred_activity', 'destination', 'budget', 'duration', 'value_score', 'user_satisfaction']
        X = df[self.feature_columns].values
        y = df['recommended_vacation'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Her destinasyon için profil oluştur
        # Özellik uzayında her destinasyonun ortalama konumunu hesaplayacağız
        # Bu, IDDFS algoritmasının "hedef durumu" için referans noktası olacak
        for destination_idx in np.unique(y_train):
            # Bu destinasyonun örneklerini filtrele
            destination_samples = X_train[y_train == destination_idx]
            
            if len(destination_samples) > 0:
                # Bu destinasyon için ortalama profil oluştur
                destination_profile = np.mean(destination_samples, axis=0)
                self.destination_profiles[destination_idx] = destination_profile
        
        # Model değerlendirme
        y_pred = []
        performance_metrics = {}
        
        # Farklı derinlikler için performans ölçümü
        depths = list(range(1, self.max_depth + 1))
        accuracies = []
        prediction_times = []
        
        for depth in depths:
            start_time = time.time()
            y_pred_depth = []
            
            for instance in X_test:
                predicted_destination = self._predict_with_iddfs(instance, max_depth=depth)
                y_pred_depth.append(predicted_destination)
            
            end_time = time.time()
            prediction_time = end_time - start_time
            
            accuracy = accuracy_score(y_test, y_pred_depth)
            
            logger.info(f"Derinlik {depth}: Doğruluk {accuracy:.4f}, Tahmin Süresi: {prediction_time:.2f} saniye")
            
            accuracies.append(accuracy)
            prediction_times.append(prediction_time)
            
            performance_metrics[depth] = {
                'accuracy': accuracy,
                'prediction_time': prediction_time
            }
            
            # En iyi derinliği seç (son iterasyon için)
            if depth == self.max_depth:
                y_pred = y_pred_depth
        
        # En iyi derinliği bul
        best_depth_idx = np.argmax(accuracies)
        best_depth = depths[best_depth_idx]
        self.max_depth = best_depth  # En iyi derinliği kaydet
        
        logger.info(f"En iyi derinlik: {best_depth}, Doğruluk: {accuracies[best_depth_idx]:.4f}")
        
        logger.info("\nTest Seti Performansı (En iyi derinlik ile):")
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Performans grafiklerini çiz
        try:
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(depths, accuracies, marker='o')
            plt.title('IDDFS Derinlik vs Doğruluk')
            plt.xlabel('Derinlik')
            plt.ylabel('Doğruluk')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(depths, prediction_times, marker='o', color='r')
            plt.title('IDDFS Derinlik vs Tahmin Süresi')
            plt.xlabel('Derinlik')
            plt.ylabel('Tahmin Süresi (saniye)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('models/iddfs_performance.png', dpi=300)
            logger.info("IDDFS performans grafiği kaydedildi: models/iddfs_performance.png")
        except Exception as e:
            logger.warning(f"Grafik oluşturma hatası: {str(e)}")
        
        logger.info("IDDFS tabanlı model eğitimi tamamlandı")
    
    def _predict_with_iddfs(self, instance, max_depth):
        """IDDFS algoritması ile tahmin yap"""
        # Kademeli olarak derinliği artırarak arama yap
        for depth in range(1, max_depth + 1):
            result = self._depth_limited_search(instance, depth)
            if result is not None:
                return result
        
        # Hiçbir derinlikte sonuç bulunamadıysa, en yakın destinasyonu döndür
        return self._find_closest_destination(instance)
    
    def _depth_limited_search(self, instance, depth):
        """Sınırlı derinlikte DFS algoritması"""
        # Bu metod, derinlik sınırlı arama yapar
        # Ana fikir: Her derinlik seviyesinde, bir önceki derinlikte bulunan en iyi durumların komşularını kontrol et
        
        # Başlangıç durumu: Tüm destinasyon adayları
        candidates = list(self.destination_profiles.keys())
        
        # Derinliğe göre aday sayısını azalt
        # Daha yüksek derinliklerde, daha fazla aday değerlendirilir
        num_candidates = max(1, int(len(candidates) / depth))
        
        # Her derinlik seviyesinde, adayları değerlendir ve en iyilerini seç
        for d in range(depth):
            scores = []
            for candidate in candidates:
                profile = self.destination_profiles[candidate]
                similarity = self._calculate_similarity(instance, profile)
                scores.append((candidate, similarity))
            
            # En iyi benzerliğe sahip adayları seç
            scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in scores[:num_candidates]]
            
            # Eğer sadece bir aday kaldıysa, onu döndür
            if len(candidates) == 1:
                return candidates[0]
        
        # Son derinlikte birden fazla aday kaldıysa, en iyi olanı döndür
        if candidates:
            scores = []
            for candidate in candidates:
                profile = self.destination_profiles[candidate]
                similarity = self._calculate_similarity(instance, profile)
                scores.append((candidate, similarity))
            
            best_candidate = max(scores, key=lambda x: x[1])[0]
            return best_candidate
        
        return None
    
    def _calculate_similarity(self, instance, profile):
        """İki profil arasındaki benzerliği hesapla (ters öklid mesafesi)"""
        distance = np.sqrt(np.sum((instance - profile) ** 2))
        # Mesafe ne kadar küçükse, benzerlik o kadar büyük
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def _find_closest_destination(self, instance):
        """En yakın destinasyonu bul"""
        best_destination = None
        best_similarity = -1
        
        for destination, profile in self.destination_profiles.items():
            similarity = self._calculate_similarity(instance, profile)
            if similarity > best_similarity:
                best_similarity = similarity
                best_destination = destination
        
        return best_destination
    
    def predict(self, user_preferences):
        """Kullanıcı tercihleri için öneri yap"""
        try:
            # Kullanıcı tercihlerini ön işle
            features = {}
            
            # Kategorik değişkenler için encoding
            for col in self.label_encoders:
                if col in user_preferences:
                    features[col] = self.label_encoders[col].transform([str(user_preferences[col])])[0]
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı!")
                    return None
            
            # Sayısal değişkenler
            numerical_data = {}
            for col in ['budget', 'duration']:
                if col in user_preferences:
                    numerical_data[col] = user_preferences[col]
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı!")
                    return None
            
            # Olmayan değerleri tahmin et (value_score ve user_satisfaction)
            numerical_data['value_score'] = 3.5  # Ortalama bir değer
            numerical_data['user_satisfaction'] = 4.0  # Ortalama bir değer
            
            # Sayısal verileri ölçeklendir
            numerical_features = np.array([[
                numerical_data['budget'],
                numerical_data['duration'],
                numerical_data['value_score'],
                numerical_data['user_satisfaction']
            ]])
            scaled_numerical = self.scaler.transform(numerical_features)
            
            # Tüm özellikleri birleştir
            instance = np.zeros(len(self.feature_columns))
            feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
            
            # Kategorik değerleri yerleştir
            for col, value in features.items():
                if col in feature_indices:
                    instance[feature_indices[col]] = value
            
            # Sayısal değerleri yerleştir
            for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                if col in feature_indices:
                    instance[feature_indices[col]] = scaled_numerical[0, i]
            
            # IDDFS ile tahmin yap
            destination_idx = self._predict_with_iddfs(instance, self.max_depth)
            
            # Güven değeri hesapla
            best_profile = self.destination_profiles[destination_idx]
            confidence = self._calculate_similarity(instance, best_profile)
            
            return {'destination': destination_idx, 'confidence': float(confidence)}
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return None
    
    def save_model(self):
        """Modeli kaydet"""
        # Label encoder'ları kaydet
        joblib.dump(self.label_encoders, 'models/iddfs_label_encoders.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, 'models/iddfs_scaler.joblib')
        
        # Feature kolonlarını kaydet
        joblib.dump(self.feature_columns, 'models/iddfs_feature_columns.joblib')
        
        # Destinasyon profillerini kaydet
        joblib.dump(self.destination_profiles, 'models/iddfs_destination_profiles.joblib')
        
        # Maksimum derinliği kaydet
        joblib.dump(self.max_depth, 'models/iddfs_max_depth.joblib')
        
        logger.info("IDDFS modeli kaydedildi")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('models/iddfs_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('models/iddfs_scaler.joblib')
            
            # Feature kolonlarını yükle
            self.feature_columns = joblib.load('models/iddfs_feature_columns.joblib')
            
            # Destinasyon profillerini yükle
            self.destination_profiles = joblib.load('models/iddfs_destination_profiles.joblib')
            
            # Maksimum derinliği yükle
            self.max_depth = joblib.load('models/iddfs_max_depth.joblib')
            
            logger.info("IDDFS modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
