import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class KNNVacationRecommender:
    def __init__(self):
        self.knn_model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.feature_weights = None
        
    def train(self, df):
        """K-En Yakın Komşu (KNN) modelini eğit"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        
        # Eğitim için gerekli dizinleri oluştur
        os.makedirs("models", exist_ok=True)
        
        # Veri setini hazırla
        X = df.drop('destination', axis=1)
        y = df['destination']
        
        # Kategorik değişkenleri encode et
        categorical_cols = ['season', 'preferred_activity']
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Sayısal değişkenleri ölçeklendir
        numerical_cols = ['budget', 'duration', 'value_score', 'user_satisfaction']
        self.scaler = StandardScaler()
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        # Özellik sütunlarını kaydet
        self.feature_columns = X.columns.tolist()
        
        # Özellik ağırlıkları - kullanıcı memnuniyetini daha fazla önemse
        feature_weights = {}
        for col in self.feature_columns:
            if col == 'user_satisfaction':
                feature_weights[col] = 2.0  # Kullanıcı memnuniyeti çok daha önemli
            elif col == 'value_score':
                feature_weights[col] = 1.5  # Değer skoru da önemli
            elif col == 'budget':
                feature_weights[col] = 1.3  # Bütçe de önemli
            else:
                feature_weights[col] = 1.0  # Diğer özellikler standart ağırlık
        
        # Ağırlıklandırılmış özellikler oluştur
        X_weighted = X.copy()
        for col, weight in feature_weights.items():
            X_weighted[col] = X_weighted[col] * weight
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)
        
        # K değerini belirlemek için elbow yöntemi
        error_rates = []
        candidate_k_values = range(1, 31, 2)  # 1'den 30'a kadar tek sayılar
        
        for k in candidate_k_values:
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train, y_train)
            y_pred = knn_model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            error_rates.append(1 - accuracy)
        
        # K değeri grafiğini çiz
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(candidate_k_values, error_rates, marker='o', markersize=8, linewidth=2)
            plt.title('K Değeri ve Hata Oranı İlişkisi')
            plt.xlabel('K Değeri')
            plt.ylabel('Hata Oranı')
            plt.grid(True)
            plt.savefig('models/knn_elbow_plot.png', dpi=300, bbox_inches='tight')
            logger.info("KNN elbow grafiği kaydedildi: models/knn_elbow_plot.png")
        except Exception as e:
            logger.warning(f"Elbow grafik oluşturma hatası: {str(e)}")
        
        # En iyi k değerini bul (en düşük hata oranına sahip)
        optimal_k = candidate_k_values[np.argmin(error_rates)]
        logger.info(f"Optimal K değeri: {optimal_k}")
        
        # Hiperparametre grid'i - daha kapsamlı arama
        param_grid = {
            'n_neighbors': [max(1, optimal_k - 4), max(1, optimal_k - 2), optimal_k, min(30, optimal_k + 2), min(30, optimal_k + 4)],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
            'p': [1, 2, 3],  # p=1 for manhattan, p=2 for euclidean, p=3 for minkowski
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        # Base model
        base_knn = KNeighborsClassifier()
        
        # GridSearchCV
        logger.info("Hiperparametre optimizasyonu başlıyor...")
        grid_search = GridSearchCV(
            estimator=base_knn,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        # En iyi modeli al
        self.knn_model = grid_search.best_estimator_
        
        # Özellik ağırlıklarını kaydet (tahmin sırasında kullanmak için)
        self.feature_weights = feature_weights
        
        # Model performansını değerlendir
        logger.info(f"\nEn iyi parametreler: {grid_search.best_params_}")
        logger.info(f"En iyi cross-validation skoru: {grid_search.best_score_:.4f}")
        
        # Test seti üzerinde değerlendir
        y_pred = self.knn_model.predict(X_test)
        
        logger.info("\nTest Seti Performansı:")
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        logger.info("KNN model eğitimi tamamlandı")
    
    def predict(self, user_preferences):
        """Kullanıcı tercihleri için öneri yap"""
        if self.knn_model is None:
            logger.error("Model eğitilmemiş!")
            return None
        
        try:
            # Kullanıcı tercihlerini ön işle
            features = {}
            
            # Kategorik değişkenler için encoding
            for col in self.label_encoders:
                if col in user_preferences:
                    try:
                        # Eğer etiket daha önce görülmemişse, en yakın etiketi bul
                        if str(user_preferences[col]) not in self.label_encoders[col].classes_:
                            logger.warning(f"{col} için '{user_preferences[col]}' etiketi eğitim verilerinde bulunmuyor.")
                            # Varsayılan değer kullan
                            if col == 'season':
                                features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                            elif col == 'preferred_activity':
                                features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                            else:
                                # Rastgele bir sınıf seç (ilk sınıfı)
                                features[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])[0]
                        else:
                            features[col] = self.label_encoders[col].transform([str(user_preferences[col])])[0]
                    except ValueError as e:
                        logger.warning(f"{col} için encoding hatası: {str(e)}")
                        # Varsayılan değer kullan
                        if col == 'season':
                            features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                        elif col == 'preferred_activity':
                            features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                        else:
                            # Rastgele bir sınıf seç (ilk sınıfı)
                            features[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])[0]
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı! Varsayılan değer kullanılıyor.")
                    # Varsayılan değerler kullan
                    if col == 'season':
                        try:
                            features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                        except:
                            features[col] = 0
                    elif col == 'preferred_activity':
                        try:
                            features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                        except:
                            features[col] = 0
                    else:
                        features[col] = 0
            
            # Sayısal değişkenler
            numerical_data = {}
            for col in ['budget', 'duration']:
                if col in user_preferences:
                    numerical_data[col] = float(user_preferences[col])  # Açıkça float'a dönüştür
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı! Varsayılan değer kullanılıyor.")
                    # Varsayılan değerler kullan
                    if col == 'budget':
                        numerical_data[col] = 10000.0
                    elif col == 'duration':
                        numerical_data[col] = 7.0
            
            # Olmayan değerleri tahmin et (value_score ve user_satisfaction)
            numerical_data['value_score'] = 3.5  # Ortalama bir değer
            numerical_data['user_satisfaction'] = 4.0  # Ortalama bir değer
            
            # Sayısal verileri ölçeklendir
            numerical_features = np.array([[
                numerical_data['budget'],
                numerical_data['duration'],
                numerical_data['value_score'],
                numerical_data['user_satisfaction']
            ]], dtype=np.float64)  # Veri tipini açıkça belirt
            
            # Feature names ekleyerek uyarıları önle
            numerical_df = pd.DataFrame(numerical_features, 
                                      columns=['budget', 'duration', 'value_score', 'user_satisfaction'])
            
            try:
                scaled_numerical = self.scaler.transform(numerical_df)
            except Exception as e:
                logger.error(f"Ölçeklendirme hatası: {str(e)}")
                # Ölçeklendirme yapılamazsa, ham verileri kullan
                scaled_numerical = numerical_features
            
            # Tüm özellikleri birleştir - veri tipini açıkça belirt
            X_pred = np.zeros((1, len(self.feature_columns)), dtype=np.float64)
            feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
            
            # Kategorik değerleri yerleştir
            for col, value in features.items():
                if col in feature_indices:
                    X_pred[0, feature_indices[col]] = float(value)  # float'a dönüştür
            
            # Sayısal değerleri yerleştir
            for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                if col in feature_indices:
                    X_pred[0, feature_indices[col]] = float(scaled_numerical[0, i])  # float'a dönüştür
            
            # Özellik ağırlıklandırma - kullanıcı memnuniyetini daha fazla önemse
            if hasattr(self, 'feature_weights') and self.feature_weights:
                for col, weight in self.feature_weights.items():
                    if col in feature_indices:
                        # Kullanıcı memnuniyeti ağırlığını artır
                        if col == 'user_satisfaction':
                            X_pred[0, feature_indices[col]] *= float(weight) * 1.5  # Ek ağırlık
                        else:
                            X_pred[0, feature_indices[col]] *= float(weight)  # float'a dönüştür
            
            # X_pred'in bellek düzenini garanti et
            X_pred = np.ascontiguousarray(X_pred)
            
            # Tahmin yap
            try:
                destination = self.knn_model.predict(X_pred)[0]
                
                # Komşuluk mesafeleri ve en yakın komşular hakkında bilgi
                distances, indices = self.knn_model.kneighbors(X_pred)
                
                # Geliştirilmiş güven hesabı: Mesafelerin ters ağırlıklı ortalaması
                if len(distances[0]) > 0:
                    # Mesafelerin tersini al (daha yakın = daha yüksek değer)
                    inverse_distances = 1.0 / (distances[0] + 1e-6)
                    # Normalize et
                    weights = inverse_distances / np.sum(inverse_distances)
                    
                    # Ağırlıklı güven değeri
                    confidence = np.sum(weights) / len(weights)
                    # Sigmoid fonksiyonu ile 0-1 aralığına normalize et
                    confidence = 1.0 / (1.0 + np.exp(-5 * (confidence - 0.5)))
                else:
                    confidence = 0.5
                
                return {'destination': destination, 'confidence': float(confidence)}
            except Exception as e:
                logger.error(f"KNN tahmin hatası: {str(e)}")
                # Fallback olarak en yakın komşuları manuel olarak bul
                try:
                    # Eğitim verilerini al
                    X_train = self.knn_model._fit_X
                    y_train = self.knn_model._y
                    
                    # Öklid mesafelerini hesapla
                    distances = np.sqrt(np.sum((X_train - X_pred) ** 2, axis=1))
                    
                    # En yakın k komşuyu bul
                    k = min(5, len(distances))
                    nearest_indices = np.argsort(distances)[:k]
                    
                    # En sık görülen sınıfı bul
                    from collections import Counter
                    nearest_classes = [y_train[i] for i in nearest_indices]
                    most_common = Counter(nearest_classes).most_common(1)[0][0]
                    
                    # Güven değeri hesapla
                    confidence = 1.0 / (1.0 + np.mean(distances[nearest_indices]))
                    
                    return {'destination': most_common, 'confidence': float(confidence)}
                except Exception as nested_e:
                    logger.error(f"Fallback tahmin hatası: {str(nested_e)}")
                    # Son çare olarak rastgele bir destinasyon döndür
                    import random
                    destinations = ['Antalya', 'Bodrum', 'İstanbul', 'Kapadokya', 'Uludağ']
                    return {'destination': random.choice(destinations), 'confidence': 0.3}
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def predict_top_n(self, user_preferences, top_n=5):
        """
        Kullanıcı tercihlerine göre en yüksek güvenilirlik skoruna sahip top_n tatil önerisini döndürür.
        
        Args:
            user_preferences (dict): Kullanıcı tercihleri
            top_n (int): Döndürülecek öneri sayısı
            
        Returns:
            list: En yüksek skorlu top_n tatil önerisi
        """
        if self.knn_model is None:
            logger.error("Model eğitilmemiş!")
            return None
        
        try:
            # Kullanıcı tercihlerini ön işle
            features = {}
            
            # Kategorik değişkenler için encoding
            for col in self.label_encoders:
                if col in user_preferences:
                    try:
                        features[col] = self.label_encoders[col].transform([str(user_preferences[col])])[0]
                    except ValueError as e:
                        logger.warning(f"{col} için encoding hatası: {str(e)}")
                        # Varsayılan değer kullan
                        if col == 'season':
                            features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                        elif col == 'preferred_activity':
                            features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                        else:
                            features[col] = 0
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı! Varsayılan değer kullanılıyor.")
                    # Varsayılan değerler kullan
                    if col == 'season':
                        try:
                            features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                        except:
                            features[col] = 0
                    elif col == 'preferred_activity':
                        try:
                            features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                        except:
                            features[col] = 0
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
            ]], dtype=np.float64)  # Veri tipini açıkça belirt
            
            # Feature names ekleyerek uyarıları önle
            numerical_df = pd.DataFrame(numerical_features, 
                                      columns=['budget', 'duration', 'value_score', 'user_satisfaction'])
            
            try:
                scaled_numerical = self.scaler.transform(numerical_df)
            except Exception as e:
                logger.error(f"Ölçeklendirme hatası: {str(e)}")
                # Ölçeklendirme yapılamazsa, ham verileri kullan
                scaled_numerical = numerical_features
            
            # Tüm özellikleri birleştir - veri tipini açıkça belirt
            X_pred = np.zeros((1, len(self.feature_columns)), dtype=np.float64)
            feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
            
            # Kategorik değerleri yerleştir
            for col, value in features.items():
                if col in feature_indices:
                    X_pred[0, feature_indices[col]] = float(value)  # float'a dönüştür
            
            # Sayısal değerleri yerleştir
            for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                if col in feature_indices:
                    X_pred[0, feature_indices[col]] = float(scaled_numerical[0, i])  # float'a dönüştür
            
            # Özellik ağırlıklandırma
            if hasattr(self, 'feature_weights') and self.feature_weights:
                for col, weight in self.feature_weights.items():
                    if col in feature_indices:
                        X_pred[0, feature_indices[col]] *= float(weight)  # float'a dönüştür
            
            # X_pred'in bellek düzenini garanti et
            X_pred = np.ascontiguousarray(X_pred)
            
            # En yakın komşuları bul
            distances, indices = self.knn_model.kneighbors(X_pred, n_neighbors=min(top_n * 2, len(self.knn_model._fit_X)))
            
            # Her bir komşunun sınıfını ve mesafesini al
            neighbors = []
            for i, neighbor_idx in enumerate(indices[0]):
                neighbor_class = self.knn_model._y[neighbor_idx]
                neighbor_distance = distances[0][i]
                neighbors.append((neighbor_class, neighbor_distance))
            
            # Sınıfları ve mesafeleri grupla
            class_distances = {}
            for cls, dist in neighbors:
                if cls not in class_distances:
                    class_distances[cls] = []
                class_distances[cls].append(dist)
            
            # Her sınıf için ortalama mesafeyi ve güven skorunu hesapla
            class_scores = []
            for cls, dists in class_distances.items():
                avg_dist = np.mean(dists)
                # Mesafe ne kadar küçükse, skor o kadar büyük
                confidence = 1.0 / (1.0 + avg_dist)
                class_scores.append((cls, confidence))
            
            # En yüksek skorlu top_n sınıfı seç
            class_scores.sort(key=lambda x: x[1], reverse=True)
            top_classes = class_scores[:top_n]
            
            # Sonuçları formatla
            results = []
            for cls, confidence in top_classes:
                results.append({
                    'destination': cls,
                    'confidence': float(confidence)
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Top-N tahmin hatası: {str(e)}")
            return None
    
    def calculate_costs(self, destination, duration):
        """
        Destinasyon ve süreye göre maliyet hesapla
        """
        # Basit bir maliyet hesaplama
        base_costs = {
            'İstanbul': {'hotel': 1500, 'flight': 1000},
            'Antalya': {'hotel': 1200, 'flight': 1200},
            'Bodrum': {'hotel': 1800, 'flight': 1300},
            'Kapadokya': {'hotel': 1000, 'flight': 1100},
            'Pamukkale': {'hotel': 900, 'flight': 1000},
            'Marmaris': {'hotel': 1300, 'flight': 1200},
            'Fethiye': {'hotel': 1100, 'flight': 1100},
            'Çeşme': {'hotel': 1400, 'flight': 1000},
            'Uludağ': {'hotel': 1600, 'flight': 900},
            'Alaçatı': {'hotel': 1700, 'flight': 1100}
        }
        
        # Varsayılan değerler
        default_costs = {'hotel': 1200, 'flight': 1000}
        
        # Destinasyon için maliyet bilgilerini al veya varsayılan değerleri kullan
        costs = base_costs.get(destination, default_costs)
        
        # Toplam maliyeti hesapla
        hotel_price = costs['hotel']
        flight_cost = costs['flight']
        total_cost = (hotel_price * duration) + flight_cost
        
        return {
            'hotel_price': hotel_price,
            'flight_cost': flight_cost,
            'total_cost': total_cost
        }
    
    def save_model(self):
        """Modeli kaydet"""
        import joblib
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        if self.knn_model is None:
            logger.error("Kaydedilecek model yok!")
            return
        
        # Modeli kaydet
        joblib.dump(self.knn_model, 'saved_models/knn_model.joblib')
        
        # Label encoder'ları kaydet
        joblib.dump(self.label_encoders, 'saved_models/knn_label_encoders.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, 'saved_models/knn_scaler.joblib')
        
        # Özellik ağırlıklarını kaydet
        if self.feature_weights is not None:
            joblib.dump(self.feature_weights, 'saved_models/knn_feature_weights.joblib')
        
        # Özellik sütunlarını kaydet
        if self.feature_columns is not None:
            joblib.dump(self.feature_columns, 'saved_models/knn_feature_columns.joblib')
        
        logger.info("KNN modeli kaydedildi.")
        return True
    
    def load_model(self):
        """Modeli yükle"""
        import joblib
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Modeli yükle
            self.knn_model = joblib.load('saved_models/knn_model.joblib')
            
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('saved_models/knn_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('saved_models/knn_scaler.joblib')
            
            # Özellik ağırlıklarını yükle
            try:
                self.feature_weights = joblib.load('saved_models/knn_feature_weights.joblib')
            except:
                logger.warning("Özellik ağırlıkları yüklenemedi.")
            
            # Özellik sütunlarını yükle
            try:
                self.feature_columns = joblib.load('saved_models/knn_feature_columns.joblib')
            except:
                logger.warning("Özellik sütunları yüklenemedi.")
            
            logger.info("KNN modeli başarıyla yüklendi.")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
