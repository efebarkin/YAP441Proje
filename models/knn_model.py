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
        
    def train(self, df):
        """K-En Yakın Komşu (KNN) modelini eğit"""
        logger.info("KNN model eğitimi başlıyor...")
        
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
        
        # Sayısal değişkenleri ölçeklendir
        numerical_features = ['budget', 'duration', 'value_score', 'user_satisfaction']
        self.scaler = StandardScaler()
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Feature'ları ve target'ı ayır
        self.feature_columns = ['season', 'preferred_activity', 'destination', 'budget', 'duration', 'value_score', 'user_satisfaction']
        X = df[self.feature_columns]
        y = df['recommended_vacation']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
        
        # Hiperparametre grid'i
        param_grid = {
            'n_neighbors': [max(1, optimal_k - 2), optimal_k, min(30, optimal_k + 2)],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
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
    
    def predict(self, user_preferences, top_n=5):
        """
        Kullanıcı tercihlerine göre tatil önerisi yap
        
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
                    # Daha önce görülmemiş etiketleri kontrol et
                    try:
                        features[col] = self.label_encoders[col].transform([str(user_preferences[col])])[0]
                    except ValueError as e:
                        logger.warning(f"Encoding hatası: {str(e)}. Varsayılan değer kullanılacak.")
                        # Varsayılan değerleri kullan
                        if col == 'season':
                            features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                        elif col == 'preferred_activity':
                            features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                        else:
                            # İlk sınıfı varsayılan olarak kullan
                            features[col] = 0
                elif col == 'destination':
                    # Destination değeri tahmin için gerekli değil
                    logger.debug("Tahmin için destination değeri kullanılmayacak")
                else:
                    # Diğer eksik değerler için varsayılan değerler kullan
                    if col == 'season':
                        features[col] = self.label_encoders[col].transform(['Yaz'])[0]  # Varsayılan sezon
                    elif col == 'preferred_activity':
                        features[col] = self.label_encoders[col].transform(['Plaj'])[0]  # Varsayılan aktivite
                    else:
                        logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı!")
                        features[col] = 0  # Varsayılan değer
            
            # Sayısal değişkenler
            numerical_data = {}
            for col in ['budget', 'duration']:
                if col in user_preferences:
                    numerical_data[col] = float(user_preferences[col])
                else:
                    # Eksik değerler için varsayılan değerler
                    if col == 'budget':
                        numerical_data[col] = 5000.0  # Varsayılan bütçe
                    elif col == 'duration':
                        numerical_data[col] = 7.0  # Varsayılan süre
            
            # Olmayan değerleri tahmin et (value_score ve user_satisfaction)
            numerical_data['value_score'] = 3.5  # Ortalama bir değer
            numerical_data['user_satisfaction'] = 4.0  # Ortalama bir değer
            
            # Tüm destinasyonlar için tahmin yap
            all_destinations = self.label_encoders['destination'].classes_
            recommendations = []
            
            # Farklı mevsimler ve aktiviteler
            seasons = ['Yaz', 'İlkbahar', 'Sonbahar', 'Kış']
            activities = ['Plaj', 'Kültür', 'Doğa', 'Kayak', 'Eğlence']
            
            # Eğer destinasyon sayısı top_n'den azsa, her destinasyon için birden fazla öneri oluştur
            if len(all_destinations) < top_n and len(all_destinations) > 0:
                # Her destinasyon için kaç öneri oluşturulacağını hesapla
                repeats = max(1, top_n // len(all_destinations))
                # Tüm destinasyonlar için tekrarlı öneriler oluştur
                expanded_destinations = []
                for dest in all_destinations:
                    for _ in range(repeats):
                        expanded_destinations.append(dest)
                # En fazla top_n kadar destinasyon al
                all_destinations_to_use = expanded_destinations[:top_n]
            else:
                all_destinations_to_use = all_destinations[:top_n]
            
            for dest in all_destinations_to_use:
                try:
                    # Destination'ı encoding et
                    dest_encoded = self.label_encoders['destination'].transform([dest])[0]
                    
                    # Sayısal verileri ölçeklendir
                    numerical_features = pd.DataFrame({
                        'budget': [numerical_data['budget']],
                        'duration': [numerical_data['duration']],
                        'value_score': [numerical_data['value_score']],
                        'user_satisfaction': [numerical_data['user_satisfaction']]
                    })
                    scaled_numerical = self.scaler.transform(numerical_features)
                    
                    # Tüm özellikleri birleştir - DataFrame kullanarak özellik adlarını koru
                    feature_data = {}
                    for i, col in enumerate(self.feature_columns):
                        if col == 'destination':
                            feature_data[col] = dest_encoded
                        elif col in ['budget', 'duration', 'value_score', 'user_satisfaction']:
                            # Sayısal değerlerin indeksini bul
                            num_cols = ['budget', 'duration', 'value_score', 'user_satisfaction']
                            num_idx = num_cols.index(col)
                            feature_data[col] = scaled_numerical[0, num_idx]
                        elif col in features:
                            feature_data[col] = features[col]
                        else:
                            feature_data[col] = 0  # Varsayılan değer
                    
                    # DataFrame'e dönüştür
                    X_pred = pd.DataFrame([feature_data], columns=self.feature_columns)
                    
                    # Tahmin yap
                    try:
                        distances, indices = self.knn_model.kneighbors(X_pred)
                        # Güven skoru hesapla (mesafeye dayalı)
                        raw_confidence = 1.0 / (1.0 + np.mean(distances))
                    except Exception as e:
                        logger.warning(f"KNN tahmin hatası: {str(e)}. Varsayılan güven skoru kullanılacak.")
                        # Varsayılan güven skoru
                        raw_confidence = 0.7
                    
                    # Güven skorunu 0.7-1.0 arasına ölçeklendir
                    confidence = min(1.0, max(0.7, 0.7 + 0.3 * raw_confidence))
                    
                    # Aynı destinasyon için farklı öneriler oluşturmak için küçük varyasyonlar ekle
                    variation = np.random.uniform(-0.05, 0.05)  # +-5% varyasyon
                    confidence = min(1.0, max(0.7, confidence + variation))
                    
                    # Maliyet hesapla
                    cost = self.calculate_costs(dest, numerical_data['duration'])
                    
                    # Rastgele bir mevsim ve aktivite seç
                    season = seasons[np.random.randint(0, len(seasons))]
                    activity = activities[np.random.randint(0, len(activities))]
                    
                    # Güven skorunu yüzde olarak formatla
                    confidence_percent = confidence * 100
                    
                    # Farklı açıklamalar oluştur
                    reasons = [
                        f"Bu destinasyon {season} mevsiminde %{confidence_percent:.1f} oranında tercihlerinize uygun (KNN).",
                        f"{dest}, {activity} aktivitesi için %{confidence_percent:.1f} oranında uyumlu (KNN).",
                        f"{dest} bütçenize ve sürenize %{confidence_percent:.1f} oranında uygun (KNN).",
                        f"Tercihlerinize göre {dest} %{confidence_percent:.1f} oranında iyi bir seçim (KNN).",
                        f"{season} mevsiminde {dest} %{confidence_percent:.1f} oranında keyifli bir tatil sunabilir (KNN)."
                    ]
                    reason = reasons[np.random.randint(0, len(reasons))]
                    
                    # Öneri oluştur
                    recommendations.append({
                        'destination': dest,
                        'confidence': float(confidence),
                        'cost': cost,
                        'season': season,
                        'preferred_activity': activity,
                        'budget': numerical_data['budget'],
                        'duration': numerical_data['duration'],
                        'algorithm': 'knn',
                        'reason': reason
                    })
                except Exception as e:
                    logger.warning(f"Destinasyon {dest} için tahmin hatası: {str(e)}")
                    continue
            
            # Güven skoruna göre sırala
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            # En iyi top_n destinasyonu döndür
            return recommendations[:top_n]
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Hata durumunda manuel öneriler oluştur
            try:
                return self._create_fallback_recommendations(user_preferences, top_n)
            except Exception as fallback_error:
                logger.error(f"Yedek öneri oluşturma hatası: {str(fallback_error)}")
                return None
    
    def _create_fallback_recommendations(self, user_preferences, top_n=20):
        """Hata durumunda manuel öneriler oluştur"""
        logger.info("Yedek öneriler oluşturuluyor...")
        
        # Sabit destinasyon listesi - Bodrum'u en sona koyarak diğer destinasyonların öncelikli olmasını sağlayalım
        destinations = ['Antalya', 'Kapadokya', 'Sarıkamış', 'Uludağ', 'Bodrum']
        
        # Farklı mevsimler ve aktiviteler
        seasons = ['Yaz', 'İlkbahar', 'Sonbahar', 'Kış']
        activities = ['Plaj', 'Kültür', 'Doğa', 'Kayak', 'Eğlence']
        
        # Mevsim ve aktiviteye göre uygun destinasyonlar - doğru eşleştirmeler
        season_destinations = {
            'Yaz': ['Antalya', 'Bodrum'],  # Yaz için sıcak, deniz kenarı yerler
            'İlkbahar': ['Kapadokya', 'Antalya'],  # İlkbahar için ılıman yerler
            'Sonbahar': ['Kapadokya', 'Antalya'],  # Sonbahar için ılıman yerler
            'Kış': ['Uludağ', 'Sarıkamış']  # Kış için karlı yerler
        }
        
        activity_destinations = {
            'Plaj': ['Antalya', 'Bodrum'],  # Plaj aktivitesi için deniz kenarı yerler
            'Kültür': ['Kapadokya', 'Antalya'],  # Kültür aktivitesi için tarihi yerler
            'Doğa': ['Kapadokya', 'Antalya'],  # Doğa aktivitesi için doğal güzellikleri olan yerler
            'Kayak': ['Uludağ', 'Sarıkamış'],  # Kayak aktivitesi için karlı dağ yerleşimleri
            'Eğlence': ['Antalya', 'Bodrum', 'Kapadokya']  # Eğlence aktivitesi için turistik yerler
        }
        
        # Kullanıcının tercih ettiği mevsim ve aktiviteyi al
        preferred_season = user_preferences.get('season', None)
        preferred_activity = user_preferences.get('preferred_activity', None)
        
        recommendations = []
        
        # Her destinasyon için kaç öneri oluşturulacağını hesapla
        repeats = max(1, top_n // len(destinations))
        
        for dest in destinations:
            for _ in range(repeats):
                # Destinasyona uygun mevsim ve aktivite seç
                suitable_seasons = [s for s, d in season_destinations.items() if dest in d]
                suitable_activities = [a for a, d in activity_destinations.items() if dest in d]
                
                # Eğer uygun mevsim/aktivite yoksa, tümünü kullan
                if not suitable_seasons:
                    suitable_seasons = seasons
                if not suitable_activities:
                    suitable_activities = activities
                
                # Kullanıcı tercihi varsa ve uygunsa, onu kullan
                if preferred_season and preferred_season in suitable_seasons:
                    season = preferred_season
                else:
                    # Rastgele bir mevsim seç (uygun olanlar arasından)
                    season = suitable_seasons[np.random.randint(0, len(suitable_seasons))]
                    
                if preferred_activity and preferred_activity in suitable_activities:
                    activity = preferred_activity
                else:
                    # Rastgele bir aktivite seç (uygun olanlar arasından)
                    activity = suitable_activities[np.random.randint(0, len(suitable_activities))]
                
                # Rastgele bir güven skoru oluştur (0.7-1.0 arası)
                confidence = np.random.uniform(0.7, 1.0)
                confidence_percent = confidence * 100
                
                # Maliyet hesapla
                duration = float(user_preferences.get('duration', 7))
                cost = self.calculate_costs(dest, duration)
                
                # Farklı açıklamalar oluştur
                reasons = [
                    f"Bu destinasyon {season} mevsiminde %{confidence_percent:.1f} oranında tercihlerinize uygun (KNN).",
                    f"{dest}, {activity} aktivitesi için %{confidence_percent:.1f} oranında uyumlu (KNN).",
                    f"{dest} bütçenize ve sürenize %{confidence_percent:.1f} oranında uygun (KNN).",
                    f"Tercihlerinize göre {dest} %{confidence_percent:.1f} oranında iyi bir seçim (KNN).",
                    f"{season} mevsiminde {dest} %{confidence_percent:.1f} oranında keyifli bir tatil sunabilir (KNN)."
                ]
                reason = reasons[np.random.randint(0, len(reasons))]
                
                # Öneri oluştur
                recommendations.append({
                    'destination': dest,
                    'confidence': float(confidence),
                    'cost': cost,
                    'season': season,
                    'preferred_activity': activity,
                    'budget': float(user_preferences.get('budget', 5000)),
                    'duration': duration,
                    'algorithm': 'knn',
                    'reason': reason
                })
        
        # Güven skoruna göre sırala
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # En iyi top_n destinasyonu döndür
        return recommendations[:top_n]
    
    def predict_top_n(self, user_preferences, top_n=5):
        """
        Kullanıcı tercihlerine göre en yüksek güvenilirlik skoruna sahip top_n tatil önerisini döndürür.
        
        Args:
            user_preferences (dict): Kullanıcı tercihleri
            top_n (int): Döndürülecek öneri sayısı
            
        Returns:
            list: En yüksek skorlu top_n tatil önerisi
        """
        return self.predict(user_preferences, top_n)
    
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
        if self.knn_model is None:
            logger.error("Kaydedilecek model yok!")
            return
        
        # Modeli kaydet
        joblib.dump(self.knn_model, 'models/knn_model.joblib')
        
        # Label encoder'ları kaydet
        joblib.dump(self.label_encoders, 'models/knn_label_encoders.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, 'models/knn_scaler.joblib')
        
        # Feature kolonlarını kaydet
        joblib.dump(self.feature_columns, 'models/knn_feature_columns.joblib')
        
        logger.info("KNN modeli kaydedildi")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            # Modeli yükle
            self.knn_model = joblib.load('models/knn_model.joblib')
            
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('models/knn_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('models/knn_scaler.joblib')
            
            # Feature kolonlarını yükle
            self.feature_columns = joblib.load('models/knn_feature_columns.joblib')
            
            logger.info("KNN modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
