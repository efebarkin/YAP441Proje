import pandas as pd
import numpy as np
import heapq
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
import time
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AStarVacationRecommender:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.destinations = []
        self.destination_encodings = {}
        self.destination_profiles = {}
        self.feature_weights = None  # Özellik ağırlıkları
        
    def train(self, df):
        """A* arama algoritması tabanlı tatil önerici modeli eğit"""
        logger.info("A* tabanlı model eğitimi başlıyor...")
        
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
        for destination_idx in np.unique(y_train):
            # Bu destinasyonun örneklerini filtrele
            destination_samples = X_train[y_train == destination_idx]
            
            if len(destination_samples) > 0:
                # Bu destinasyon için ortalama profil oluştur
                destination_profile = np.mean(destination_samples, axis=0)
                self.destination_profiles[destination_idx] = destination_profile
        
        # A* için özellik ağırlıklarını öğren
        self.feature_weights = self._learn_feature_weights(X_train, y_train)
        
        # Model değerlendirme
        start_time = time.time()
        y_pred = []
        
        for instance in X_test:
            predicted_destination = self._predict_with_astar(instance)
            y_pred.append(predicted_destination)
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test Seti Doğruluk: {accuracy:.4f}, Tahmin Süresi: {prediction_time:.2f} saniye")
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Özellik ağırlıkları grafiği
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(self.feature_columns, self.feature_weights)
            plt.title('A* Algoritması Özellik Ağırlıkları')
            plt.xlabel('Özellikler')
            plt.ylabel('Ağırlık')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('models/astar_feature_weights.png', dpi=300)
            logger.info("A* özellik ağırlıkları grafiği kaydedildi: models/astar_feature_weights.png")
        except Exception as e:
            logger.warning(f"Grafik oluşturma hatası: {str(e)}")
        
        logger.info("A* tabanlı model eğitimi tamamlandı")
    
    def _learn_feature_weights(self, X, y, iterations=100, learning_rate=0.01):
        logger.info("Özellik ağırlıkları öğreniliyor...")
        
        # Başlangıçta daha iyimser ağırlıklar - eşit yerine veri tabanlı başlangıç
        # Özellik önem derecelerini basit korelasyon ile tahmin et
        weights = np.ones(X.shape[1])
        
        # Her özellik için korelasyon hesapla
        for j in range(X.shape[1]):
            feature_corrs = []
            for dest in np.unique(y):
                # Bu destinasyona ait örnekleri seç
                dest_samples = X[y == dest, j]
                # Diğer destinasyonlara ait örnekleri seç
                other_samples = X[y != dest, j]
                
                if len(dest_samples) > 0 and len(other_samples) > 0:
                    # Ortalama farkı hesapla (ayrıştırıcılık)
                    mean_diff = abs(np.mean(dest_samples) - np.mean(other_samples))
                    feature_corrs.append(mean_diff)
            
            if feature_corrs:
                # Bu özelliğin ortalama ayrıştırıcılığı
                weights[j] = np.mean(feature_corrs) + 0.1  # 0'dan kaçınmak için offset ekle
        
        # Ağırlıkları normalize et
        weights = weights / np.sum(weights)
        
        # Adaptif öğrenme oranı ve momentum ekle
        momentum = 0.9
        prev_updates = np.zeros_like(weights)
        adaptive_lr = learning_rate
        
        # Her iterasyonda ağırlıkları güncelle
        best_accuracy = 0
        best_weights = weights.copy()
        patience = 10  # Erken durdurma için sabır
        no_improve_count = 0
        
        for iteration in range(iterations):
            correct_count = 0
            weight_updates = np.zeros_like(weights)
            
            for i, instance in enumerate(X):
                predicted = self._predict_with_weights(instance, weights)
                true_label = y[i]
                
                if predicted == true_label:
                    correct_count += 1
                else:
                    # True label profili
                    true_profile = self.destination_profiles[true_label]
                    
                    # Predicted label profili
                    pred_profile = self.destination_profiles[predicted]
                    
                    # Ağırlık güncellemelerini hesapla - iyileştirilmiş formül
                    for j in range(len(weights)):
                        # Doğru sınıf özelliğine yaklaştır, yanlış sınıf özelliğinden uzaklaştır
                        # Karesel fark yerine mutlak fark kullan - aşırı cezalandırmayı önle
                        true_diff = abs(instance[j] - true_profile[j])
                        pred_diff = abs(instance[j] - pred_profile[j])
                        
                        # Daha iyimser güncelleme: Doğru tahmin için ödüllendirme faktörü ekle
                        update = (pred_diff - true_diff) * (1 + 0.2 * weights[j])
                        weight_updates[j] += update
            
            # Momentum ile ağırlıkları güncelle
            current_updates = adaptive_lr * weight_updates / len(X)
            updates_with_momentum = current_updates + momentum * prev_updates
            weights += updates_with_momentum
            prev_updates = updates_with_momentum
            
            # Ağırlıkları normalize et
            weights = np.abs(weights)  # Negatif değerleri engelle
            weights = weights / np.sum(weights)  # Toplamı 1 yap
            
            # Doğruluk hesapla
            accuracy = correct_count / len(X)
            
            # En iyi ağırlıkları sakla
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adaptif öğrenme oranını güncelle
            if no_improve_count > 0:
                adaptive_lr *= 0.95  # Öğrenme oranını azalt
            
            # Erken durdurma
            if no_improve_count >= patience:
                logger.info(f"İterasyon {iteration + 1}/{iterations} - Erken durdurma (En iyi doğruluk: {best_accuracy:.4f})")
                break
            
            # İlerlemeyi logla
            if (iteration + 1) % 10 == 0:
                logger.info(f"İterasyon {iteration + 1}/{iterations}, Doğruluk: {accuracy:.4f}")
        
        # En iyi ağırlıkları kullan
        weights = best_weights
        logger.info(f"Özellik ağırlıkları öğrenildi. En iyi doğruluk: {best_accuracy:.4f}")
        return weights
    
    def _predict_with_weights(self, instance, weights):
        """Verilen ağırlıkları kullanarak tahmin yap"""
        best_destination = None
        best_score = float('-inf')
        
        for destination, profile in self.destination_profiles.items():
            # A* skoru hesapla
            g_score = 0  # Başlangıç maliyeti
            h_score = self._weighted_heuristic(instance, profile, weights)
            f_score = g_score + h_score
            
            if f_score > best_score:  # A* en yüksek skoru seçer (benzerlik)
                best_score = f_score
                best_destination = destination
        
        return best_destination
    
    def _predict_with_astar(self, instance):
        """A* algoritması ile tahmin yap"""
        # Boş öncelikli kuyruk başlat (priority queue)
        open_set = []
        
        # Her destinasyon için başlangıç f-skoru hesapla ve kuyruğa ekle
        for destination, profile in self.destination_profiles.items():
            g_score = 0  # Başlangıç maliyeti
            h_score = self._weighted_heuristic(instance, profile, self.feature_weights)
            f_score = g_score + h_score
            
            # Negatif f_score ekle çünkü heapq en küçüğü önceliklendirir, 
            # ama biz en büyük benzerliği istiyoruz
            heapq.heappush(open_set, (-f_score, destination))
        
        # Hiç destinasyon yoksa None döndür
        if not open_set:
            return None
        
        # En iyi f-skora sahip destinasyonu döndür
        best_score, best_destination = heapq.heappop(open_set)
        return best_destination
    
    def _weighted_heuristic(self, instance, destination_profile, weights):
        similarity = 0.0
        
        # Kategorik ve sayısal özellikleri ayrı değerlendir
        categorical_indices = [i for i, col in enumerate(self.feature_columns) 
                            if col in ['season', 'preferred_activity', 'destination']]
        numerical_indices = [i for i, col in enumerate(self.feature_columns) 
                            if col in ['budget', 'duration', 'value_score', 'user_satisfaction']]
        
        # Kategorik özellikler için tam eşleşme ödülü
        for i in categorical_indices:
            if instance[i] == destination_profile[i]:
                # Tam eşleşme için bonus
                similarity += 2.0 * weights[i]
            else:
                # Eşleşmeme için daha az ceza
                diff = abs(instance[i] - destination_profile[i])
                similarity -= diff * weights[i] * 0.5
        
        # Sayısal özellikler için mesafe tabanlı benzerlik
        for i in numerical_indices:
            # Mesafe hesapla
            diff = abs(instance[i] - destination_profile[i])
            
            # Normalize edilmiş veri için uygun bir eşik
            threshold = 0.5
            
            # Eşik altındaki farklar için daha az ceza
            if diff < threshold:
                similarity -= (diff**2) * weights[i]
            else:
                # Eşik üstü için lineer ceza - aşırı cezalandırmayı önle
                similarity -= (threshold**2 + (diff - threshold)) * weights[i]
        
        # Bonus faktörü ekle - daha iyimser heuristic için
        optimism_factor = 1.2
        if similarity > 0:
            similarity *= optimism_factor
        
        return similarity
    
    def predict(self, user_preferences):
        """Kullanıcı tercihleri için öneri yap"""
        try:
            # Kullanıcı tercihlerini ön işle
            features = {}
            
            # Kategorik değişkenler için encoding
            for col in self.label_encoders:
                if col == 'destination':
                    # Destination tahmin edilecek değer olduğu için kullanıcı tercihlerinde olmamalı
                    # Varsayılan olarak ilk destinasyonu kullan (sadece placeholder olarak)
                    features[col] = 0
                elif col in user_preferences:
                    features[col] = self.label_encoders[col].transform([str(user_preferences[col])])[0]
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı! Varsayılan değer kullanılıyor.")
                    # Varsayılan değerler kullan
                    if col == 'season':
                        features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                    elif col == 'preferred_activity':
                        features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                    else:
                        features[col] = 0
            
            # Sayısal değişkenler
            numerical_data = {}
            for col in ['budget', 'duration']:
                if col in user_preferences:
                    numerical_data[col] = user_preferences[col]
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı! Varsayılan değer kullanılıyor.")
                    # Varsayılan değerler kullan
                    if col == 'budget':
                        numerical_data[col] = 10000
                    elif col == 'duration':
                        numerical_data[col] = 7
            
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
            
            # A* ile tahmin yap
            destination_idx = self._predict_with_astar(instance)
            
            if destination_idx is None:
                logger.warning("A* algoritması bir destinasyon önerisi bulamadı.")
                return None
            
            # Güven değeri hesapla (sezgisel)
            best_profile = self.destination_profiles[destination_idx]
            raw_score = self._weighted_heuristic(instance, best_profile, self.feature_weights)
            # Normalize et
            confidence = 1.0 / (1.0 + np.exp(-raw_score * 0.5))
            
            # Destinasyon adını bul
            destination_name = None
            for name, idx in self.destination_encodings.items():
                if idx == destination_idx:
                    destination_name = name
                    break
            
            return {
                'destination': destination_idx,
                'destination_name': destination_name,
                'confidence': float(confidence)
            }
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return None
    
    def save_model(self):
        """Modeli kaydet"""
        # Label encoder'ları kaydet
        joblib.dump(self.label_encoders, 'models/astar_label_encoders.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, 'models/astar_scaler.joblib')
        
        # Feature kolonlarını kaydet
        joblib.dump(self.feature_columns, 'models/astar_feature_columns.joblib')
        
        # Destinasyon profillerini kaydet
        joblib.dump(self.destination_profiles, 'models/astar_destination_profiles.joblib')
        
        # Özellik ağırlıklarını kaydet
        joblib.dump(self.feature_weights, 'models/astar_feature_weights.joblib')
        
        logger.info("A* modeli kaydedildi")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('models/astar_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('models/astar_scaler.joblib')
            
            # Feature kolonlarını yükle
            self.feature_columns = joblib.load('models/astar_feature_columns.joblib')
            
            # Destinasyon profillerini yükle
            self.destination_profiles = joblib.load('models/astar_destination_profiles.joblib')
            
            # Özellik ağırlıklarını yükle
            self.feature_weights = joblib.load('models/astar_feature_weights.joblib')
            
            logger.info("A* modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
