import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from sklearn import tree
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DecisionTreeVacationRecommender:
    def __init__(self):
        self.dt_model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        
    def train(self, df):
        """Karar Ağacı modelini eğit"""
        logger.info("Karar Ağacı model eğitimi başlıyor...")
        
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
        
        # Hiperparametre grid'i
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced']
        }
        
        # Base model
        base_dt = DecisionTreeClassifier(random_state=42)
        
        # GridSearchCV
        logger.info("Hiperparametre optimizasyonu başlıyor...")
        grid_search = GridSearchCV(
            estimator=base_dt,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        # En iyi modeli al
        self.dt_model = grid_search.best_estimator_
        
        # Model performansını değerlendir
        logger.info(f"\nEn iyi parametreler: {grid_search.best_params_}")
        logger.info(f"En iyi cross-validation skoru: {grid_search.best_score_:.4f}")
        
        # Test seti üzerinde değerlendir
        y_pred = self.dt_model.predict(X_test)
        
        logger.info("\nTest Seti Performansı:")
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        logger.info(f"\n{feature_importance}")

        # Ağaç yapısını görselleştir (eğitim için)
        try:
            plt.figure(figsize=(20, 10))
            tree.plot_tree(self.dt_model, 
                          feature_names=self.feature_columns,
                          filled=True, 
                          max_depth=3,  # Sadece ilk 3 seviyeyi göster
                          fontsize=10)
            plt.savefig('models/decision_tree_structure.png', dpi=300, bbox_inches='tight')
            logger.info("Karar ağacı yapısı kaydedildi: models/decision_tree_structure.png")
        except Exception as e:
            logger.warning(f"Ağaç görselleştirme hatası: {str(e)}")
        
        logger.info("Karar Ağacı model eğitimi tamamlandı")
    
    def predict(self, user_preferences):
        """Kullanıcı tercihleri için öneri yap"""
        if self.dt_model is None:
            logger.error("Model eğitilmemiş!")
            return None
        
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
            X_pred = np.zeros((1, len(self.feature_columns)))
            feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
            
            # Kategorik değerleri yerleştir
            for col, value in features.items():
                if col in feature_indices:
                    X_pred[0, feature_indices[col]] = value
            
            # Sayısal değerleri yerleştir
            for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                if col in feature_indices:
                    X_pred[0, feature_indices[col]] = scaled_numerical[0, i]
            
            # Tahmin yap
            destination = self.dt_model.predict(X_pred)[0]
            
            # Olasılıkları al
            probabilities = self.dt_model.predict_proba(X_pred)[0]
            confidence = max(probabilities)
            
            return {'destination': destination, 'confidence': float(confidence)}
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
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
        if self.dt_model is None:
            logger.error("Model eğitilmemiş!")
            return []
        
        try:
            # Kullanıcı tercihlerini ön işle
            features = {}
            
            # Kategorik değişkenler için encoding
            season = user_preferences.get('season')
            preferred_activity = user_preferences.get('preferred_activity')
            destination = user_preferences.get('destination', None)
            
            if season and 'season' in self.label_encoders:
                features['season'] = self.label_encoders['season'].transform([str(season)])[0]
            else:
                logger.warning(f"season kullanıcı tercihlerinde bulunamadı veya label encoder yok!")
                return []
                
            if preferred_activity and 'preferred_activity' in self.label_encoders:
                features['preferred_activity'] = self.label_encoders['preferred_activity'].transform([str(preferred_activity)])[0]
            else:
                logger.warning(f"preferred_activity kullanıcı tercihlerinde bulunamadı veya label encoder yok!")
                return []
            
            # Destination belirtilmişse ekle, belirtilmemişse tüm destinasyonlar için tahmin yapacağız
            if destination and 'destination' in self.label_encoders:
                features['destination'] = self.label_encoders['destination'].transform([str(destination)])[0]
            
            # Sayısal değişkenler
            numerical_data = {}
            for col in ['budget', 'duration']:
                if col in user_preferences:
                    numerical_data[col] = user_preferences[col]
                else:
                    logger.warning(f"{col} kullanıcı tercihlerinde bulunamadı!")
                    return []
            
            # Olmayan değerleri tahmin et (value_score ve user_satisfaction)
            numerical_data['value_score'] = 3.5  # Ortalama bir değer
            numerical_data['user_satisfaction'] = 4.0  # Ortalama bir değer
            
            # Tüm destinasyonlar için tahmin yap
            all_destinations = self.label_encoders['destination'].classes_
            recommendations = []
            
            for dest in all_destinations:
                # Destination'ı encoding et
                dest_encoded = self.label_encoders['destination'].transform([dest])[0]
                
                # Tüm özellikleri birleştir
                X_pred = np.zeros((1, len(self.feature_columns)))
                feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
                
                # Kategorik değerleri yerleştir
                for col, value in features.items():
                    if col in feature_indices:
                        X_pred[0, feature_indices[col]] = value
                
                # Destination değerini yerleştir
                X_pred[0, feature_indices['destination']] = dest_encoded
                
                # Sayısal değerleri ölçeklendir
                numerical_features = np.array([[
                    numerical_data['budget'],
                    numerical_data['duration'],
                    numerical_data['value_score'],
                    numerical_data['user_satisfaction']
                ]])
                scaled_numerical = self.scaler.transform(numerical_features)
                
                # Sayısal değerleri yerleştir
                for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                    if col in feature_indices:
                        X_pred[0, feature_indices[col]] = scaled_numerical[0, i]
                
                # Tahmin yap
                probabilities = self.dt_model.predict_proba(X_pred)[0]
                confidence = probabilities[1] if len(probabilities) > 1 else 0  # Sınıf 1 (tavsiye edilir) olasılığı
                
                # Eğer güven değeri yeterince yüksekse, bu destinasyonu öneriler listesine ekle
                if confidence > 0.1:  # Minimum güven eşiği
                    # Destinasyon bilgilerini al
                    costs = self.calculate_costs(dest, user_preferences.get('duration', 7))
                    
                    recommendations.append({
                        'destination': dest,
                        'confidence': float(confidence),
                        'algorithm_confidence': float(confidence),
                        'season': season,
                        'activity': preferred_activity,
                        'costs': costs
                    })
            
            # Güven değerine göre sırala ve en iyi top_n tanesini döndür
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            return recommendations[:top_n]
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return []
    
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
        if self.dt_model is None:
            logger.error("Kaydedilecek model yok!")
            return
        
        # Modeli kaydet
        joblib.dump(self.dt_model, 'models/dt_model.joblib')
        
        # Label encoder'ları kaydet
        joblib.dump(self.label_encoders, 'models/dt_label_encoders.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, 'models/dt_scaler.joblib')
        
        # Feature kolonlarını kaydet
        joblib.dump(self.feature_columns, 'models/dt_feature_columns.joblib')
        
        logger.info("Karar Ağacı modeli kaydedildi")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            # Modeli yükle
            self.dt_model = joblib.load('models/dt_model.joblib')
            
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('models/dt_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('models/dt_scaler.joblib')
            
            # Feature kolonlarını yükle
            self.feature_columns = joblib.load('models/dt_feature_columns.joblib')
            
            logger.info("Karar Ağacı modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
