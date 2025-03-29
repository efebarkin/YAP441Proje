import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
        self.similarity_threshold = 0.6  # Benzerlik eşiği
        self.optimal_depth = None  # Optimal derinlik
        self.feature_weights = None  # Özellik ağırlıkları
        self.rf_classifier = None  # Random Forest sınıflandırıcı

    def train(self, df, max_depth=10, similarity_threshold=0.6):
        logger.info("IDDFS tabanlı model eğitimi başlıyor...")
        
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        
        columns_to_drop = ['user_id', 'hotel_price_per_night', 'flight_cost', 'total_cost']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
        
        categorical_features = ['season', 'preferred_activity', 'destination']
        
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
                if col == 'destination':
                    self.destinations = list(le.classes_)
                    self.destination_encodings = {le.classes_[i]: i for i in range(len(le.classes_))}
        
        numerical_features = ['budget', 'duration', 'value_score', 'user_satisfaction']
        self.scaler = StandardScaler()
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        self.feature_columns = ['season', 'preferred_activity', 'budget', 'duration', 'value_score', 'user_satisfaction']
        X = df[self.feature_columns].values
        y = df['destination'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for destination_idx in np.unique(y_train):
            destination_samples = X_train[y_train == destination_idx]
            
            if len(destination_samples) > 0:
                destination_profile = np.mean(destination_samples, axis=0)
                self.destination_profiles[destination_idx] = destination_profile
        
        self._learn_feature_weights(X_train, y_train)
        
        logger.info("Random Forest sınıflandırıcı eğitiliyor...")
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.rf_classifier.fit(X_train, y_train)
        
        rf_pred = self.rf_classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        logger.info(f"Random Forest doğruluğu: {rf_accuracy:.4f}")
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        best_threshold = 0.2
        best_accuracy = 0
        
        for threshold in thresholds:
            self.similarity_threshold = threshold
            y_pred = []
            
            for instance in X_test:
                predicted_destination = self._hybrid_predict(instance, max_depth=3)
                y_pred.append(predicted_destination)
            
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Eşik {threshold}: Doğruluk {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.similarity_threshold = best_threshold
        logger.info(f"Optimal benzerlik eşiği: {best_threshold}, Doğruluk: {best_accuracy:.4f}")
        
        y_pred = []
        performance_metrics = {}
        
        depths = list(range(1, self.max_depth + 1, 2))  
        accuracies = []
        prediction_times = []
        
        for depth in depths:
            start_time = time.time()
            y_pred_depth = []
            
            for instance in X_test:
                predicted_destination = self._hybrid_predict(instance, max_depth=depth)
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
        
        best_depth = depths[np.argmax(accuracies)]
        self.optimal_depth = best_depth
        logger.info(f"Optimal derinlik: {best_depth}, Doğruluk: {performance_metrics[best_depth]['accuracy']:.4f}")
        
        final_y_pred = []
        start_time = time.time()
        
        for instance in X_test:
            predicted_destination = self._hybrid_predict(instance, max_depth=self.optimal_depth)
            final_y_pred.append(predicted_destination)
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        final_accuracy = accuracy_score(y_test, final_y_pred)
        
        logger.info(f"Final Doğruluk: {final_accuracy:.4f}, Tahmin Süresi: {prediction_time/len(X_test):.2f} saniye")
        
        cm = confusion_matrix(y_test, final_y_pred)
        cr = classification_report(y_test, final_y_pred)
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{cr}")
        
        logger.info("IDDFS tabanlı model eğitimi tamamlandı")
        return self

    def _hybrid_predict(self, instance, max_depth):
        iddfs_prediction = self._predict_with_iddfs(instance, max_depth)
        
        if self.rf_classifier is not None:
            rf_prediction = self.rf_classifier.predict([instance])[0]
            rf_proba = self.rf_classifier.predict_proba([instance])[0]
            rf_confidence = np.max(rf_proba)
            
            if rf_confidence > 0.6:
                return rf_prediction
        
        return iddfs_prediction

    def predict(self, user_preferences, top_n=5):
        """
        Kullanıcı tercihlerine göre tatil destinasyonu öner
        """
        try:
            # Kullanıcı tercihlerini işle
            instance = self._preprocess_user_preferences(user_preferences)
            
            # Destination değerini kullanıcı tercihlerinden çıkar
            # Eğer destination varsa, bu değeri kullanma
            if 'destination' in user_preferences:
                logger.info("Tahmin için kullanıcı tercihlerinden destination değeri çıkarıldı")
            
            # IDDFS ile tahmin yap
            results = self._predict_with_iddfs_multiple(instance, top_n)
            
            # Sonuçları zenginleştir
            for result in results:
                result['season'] = user_preferences.get('season', 'Summer')
                result['preferred_activity'] = user_preferences.get('preferred_activity', 'Beach')
                result['budget'] = user_preferences.get('budget', 5000.0)
                result['duration'] = user_preferences.get('duration', 7.0)
            
            return results
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def save_model(self):
        joblib.dump(self.label_encoders, 'models/iddfs_label_encoders.joblib')
        joblib.dump(self.scaler, 'models/iddfs_scaler.joblib')
        joblib.dump(self.destination_profiles, 'models/iddfs_destination_profiles.joblib')
        joblib.dump(self.feature_columns, 'models/iddfs_feature_columns.joblib')
        joblib.dump(self.max_depth, 'models/iddfs_max_depth.joblib')
        joblib.dump(self.optimal_depth, 'models/iddfs_optimal_depth.joblib')
        joblib.dump(self.similarity_threshold, 'models/iddfs_similarity_threshold.joblib')
        joblib.dump(self.feature_weights, 'models/iddfs_feature_weights.joblib')
        
        if self.rf_classifier is not None:
            joblib.dump(self.rf_classifier, 'models/iddfs_rf_classifier.joblib')
        
        logger.info("IDDFS modeli kaydedildi")
    
    def load_model(self):
        try:
            self.label_encoders = joblib.load('models/iddfs_label_encoders.joblib')
            self.scaler = joblib.load('models/iddfs_scaler.joblib')
            self.destination_profiles = joblib.load('models/iddfs_destination_profiles.joblib')
            self.feature_columns = joblib.load('models/iddfs_feature_columns.joblib')
            self.max_depth = joblib.load('models/iddfs_max_depth.joblib')
            self.optimal_depth = joblib.load('models/iddfs_optimal_depth.joblib')
            self.similarity_threshold = joblib.load('models/iddfs_similarity_threshold.joblib')
            self.feature_weights = joblib.load('models/iddfs_feature_weights.joblib')
            
            # Destination encodings ve destinations değişkenlerini oluştur
            self.destination_encodings = {}
            self.destinations = []
            
            if 'destination' in self.label_encoders:
                le = self.label_encoders['destination']
                self.destinations = list(le.classes_)
                self.destination_encodings = {le.classes_[i]: i for i in range(len(le.classes_))}
                logger.info(f"Destinasyon listesi ve kodlamaları oluşturuldu: {len(self.destinations)} destinasyon")
            
            try:
                self.rf_classifier = joblib.load('models/iddfs_rf_classifier.joblib')
                logger.info("Random Forest sınıflandırıcı yüklendi")
            except Exception as e:
                logger.warning(f"Random Forest sınıflandırıcı yüklenemedi: {str(e)}")
                self.rf_classifier = None
            
            logger.info("IDDFS modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _learn_feature_weights(self, X, y):
        logger.info("Özellik ağırlıkları öğreniliyor...")
        
        weights = np.ones(len(self.feature_columns))
        learning_rate = 0.01
        max_iterations = 100
        patience = 5
        stagnation_count = 0
        best_accuracy = 0
        
        for iteration in range(max_iterations):
            weights = weights / np.sum(weights)
            self.feature_weights = weights
            
            y_pred = []
            for instance in X:
                predicted = self._predict_with_weights(instance, weights)
                y_pred.append(predicted)
            
            accuracy = accuracy_score(y, y_pred)
            
            if iteration % 10 == 0:
                logger.info(f"İterasyon {iteration}/{max_iterations}, Doğruluk: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count >= patience:
                logger.info(f"İterasyon {iteration}/{max_iterations} - Erken durdurma (En iyi doğruluk: {best_accuracy:.4f})")
                break
            
            for i, (pred, true) in enumerate(zip(y_pred, y)):
                if pred == true:
                    weights += learning_rate * X[i]
                else:
                    weights -= learning_rate * X[i]
            
            weights = np.maximum(0.1, weights)
        
        self.feature_weights = best_weights
        logger.info(f"Özellik ağırlıkları öğrenildi: {self.feature_weights}")
        logger.info(f"En iyi doğruluk: {best_accuracy:.4f}")
        
        return best_weights
    
    def _predict_with_weights(self, instance, weights):
        best_destination = None
        best_similarity = -1
        
        for destination, profile in self.destination_profiles.items():
            similarity = self._calculate_similarity_with_weights(instance, profile, weights)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_destination = destination
        
        return best_destination

    def _calculate_similarity(self, instance, profile):
        if self.feature_weights is None:
            weights = np.ones(len(self.feature_columns))
        else:
            weights = self.feature_weights
        
        return self._calculate_similarity_with_weights(instance, profile, weights)
    
    def _calculate_similarity_with_weights(self, instance, profile, weights):
        weighted_instance = instance * weights
        weighted_profile = profile * weights
        
        dot_product = np.dot(weighted_instance, weighted_profile)
        norm_instance = np.linalg.norm(weighted_instance)
        norm_profile = np.linalg.norm(weighted_profile)
        
        if norm_instance == 0 or norm_profile == 0:
            return 0.0
        
        cosine_similarity = dot_product / (norm_instance * norm_profile)
        
        distance = np.sqrt(np.sum(((instance - profile) * weights) ** 2))
        distance_similarity = 1.0 / (1.0 + distance)
        
        combined_similarity = 0.7 * cosine_similarity + 0.3 * distance_similarity
        
        return max(0.0, min(1.0, combined_similarity))  

    def _find_closest_destination(self, instance):
        best_destination = None
        best_similarity = -1
        
        for destination, profile in self.destination_profiles.items():
            similarity = self._calculate_similarity(instance, profile)
            if similarity > best_similarity:
                best_similarity = similarity
                best_destination = destination
        
        if best_similarity >= self.similarity_threshold:
            return best_destination
        
        if best_similarity >= 0.3:  
            return best_destination
        
        return None

    def _predict_with_iddfs(self, instance, max_depth, destination_profiles=None):
        for depth in range(1, max_depth + 1):
            result = self._depth_limited_search(instance, depth, destination_profiles)
            if result is not None:
                return result
        
        closest = self._find_closest_destination(instance)
        
        if closest is None:
            if destination_profiles is not None and len(destination_profiles) > 0:
                return list(destination_profiles.keys())[0]
            elif len(self.destination_profiles) > 0:
                return list(self.destination_profiles.keys())[0]
            else:
                return 0
        
        return closest

    def _depth_limited_search(self, instance, depth, destination_profiles=None):
        if destination_profiles is None:
            destination_profiles = self.destination_profiles
        
        similarities = []
        for dest_idx, profile in destination_profiles.items():
            similarity = self._calculate_similarity(instance, profile)
            similarities.append((dest_idx, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        num_candidates = min(len(similarities), depth + 2)
        
        if depth == 1:
            if similarities:
                best_dest, best_sim = similarities[0]
                if best_sim >= self.similarity_threshold:
                    return best_dest
            return None
        
        best_candidates = similarities[:num_candidates]
        
        if best_candidates:
            best_dest, best_sim = max(best_candidates, key=lambda x: x[1])
            if best_sim >= self.similarity_threshold * (0.9 ** (depth-1)):  
                return best_dest
        
        return None

    def _preprocess_user_preferences(self, user_preferences):
        features = {}
        numerical_data = {}
        
        if 'season' in user_preferences:
            season = user_preferences['season']
            if season in self.label_encoders['season'].classes_:
                features['season'] = self.label_encoders['season'].transform([season])[0]
            else:
                logger.warning(f"Bilinmeyen sezon: {season}")
                features['season'] = 0  
        
        if 'preferred_activity' in user_preferences:
            activity = user_preferences['preferred_activity']
            if activity in self.label_encoders['preferred_activity'].classes_:
                features['preferred_activity'] = self.label_encoders['preferred_activity'].transform([activity])[0]
            else:
                logger.warning(f"Bilinmeyen aktivite: {activity}")
                features['preferred_activity'] = 0  
        
        if 'budget' in user_preferences:
            numerical_data['budget'] = float(user_preferences['budget'])
        else:
            numerical_data['budget'] = 5000.0  
        
        if 'duration' in user_preferences:
            numerical_data['duration'] = float(user_preferences['duration'])
        else:
            numerical_data['duration'] = 7.0  
        
        numerical_data['value_score'] = 0.7
        numerical_data['user_satisfaction'] = 0.8
        
        numerical_features = pd.DataFrame({
            'budget': [numerical_data['budget']],
            'duration': [numerical_data['duration']],
            'value_score': [numerical_data['value_score']],
            'user_satisfaction': [numerical_data['user_satisfaction']]
        })
        scaled_numerical = self.scaler.transform(numerical_features)
        
        instance = np.zeros(len(self.feature_columns))
        feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
        
        for col, value in features.items():
            if col in feature_indices:
                instance[feature_indices[col]] = value
        
        for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
            if col in feature_indices:
                instance[feature_indices[col]] = scaled_numerical[0, i]
        
        return instance

    def _predict_with_iddfs_multiple(self, instance, top_n):
        results = []
        
        try:
            # Önce Random Forest ile tahmin yapmayı dene
            if hasattr(self, 'rf_classifier') and self.rf_classifier is not None:
                rf_results = self._predict_with_random_forest(instance, top_n)
                if rf_results and len(rf_results) > 0:
                    # Random Forest sonuçlarına algoritma bilgisini ekle
                    for result in rf_results:
                        result['algorithm'] = 'iddfs_rf'
                    return rf_results
            
            # Random Forest başarısız olursa veya yoksa, IDDFS ile devam et
            destination_profiles_copy = self.destination_profiles.copy()
            
            if not destination_profiles_copy:
                logger.warning("Destinasyon profilleri boş!")
                return []
                
            # Tüm destinasyonları değerlendir ve en iyi top_n'i seç
            all_destinations = []
            
            for dest_idx, profile in destination_profiles_copy.items():
                try:
                    similarity = self._calculate_similarity(instance, profile)
                    confidence = min(1.0, max(0.0, similarity))
                    
                    destination_name = None
                    for name, idx in self.destination_encodings.items():
                        if idx == dest_idx:
                            destination_name = name
                            break
                    
                    # Destination_name bulunamazsa, indeks değerini kullan
                    if not destination_name:
                        destination_name = f"Destination_{dest_idx}"
                        logger.warning(f"Destinasyon adı bulunamadı, varsayılan değer kullanılıyor: {destination_name}")
                    
                    all_destinations.append({
                        'destination': destination_name,
                        'confidence': float(confidence),
                        'dest_idx': dest_idx
                    })
                except Exception as e:
                    logger.warning(f"Destinasyon {dest_idx} değerlendirme hatası: {str(e)}")
                    continue
            
            # Güven skoruna göre sırala
            all_destinations.sort(key=lambda x: x['confidence'], reverse=True)
            
            # En iyi top_n destinasyonu seç
            for i in range(min(top_n, len(all_destinations))):
                dest = all_destinations[i]
                result = {
                    'destination': dest['destination'],
                    'confidence': dest['confidence'],
                    'algorithm': 'iddfs',
                    'reason': f"IDDFS algoritması bu destinasyonu {dest['confidence']:.2f} güven skoru ile önerdi."
                }
                results.append(result)
            
            if not results:
                logger.warning("IDDFS algoritması bir destinasyon önerisi bulamadı.")
                # Son çare olarak en popüler destinasyonu öner
                if self.destinations and len(self.destinations) > 0:
                    result = {
                        'destination': self.destinations[0],
                        'confidence': 0.5,
                        'algorithm': 'iddfs_fallback',
                        'reason': "Özel bir eşleşme bulunamadı, popüler bir destinasyon öneriliyor."
                    }
                    results.append(result)
        
        except Exception as e:
            logger.error(f"IDDFS tahmin hatası: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results

    def _predict_with_random_forest(self, instance, top_n=5):
        """Random Forest modeli ile tahmin yap"""
        if not hasattr(self, 'rf_classifier') or self.rf_classifier is None:
            logger.warning("Random Forest modeli yüklenmemiş!")
            return []
        
        try:
            # Tahmin olasılıklarını al
            probabilities = self.rf_classifier.predict_proba([instance])[0]
            
            # En yüksek olasılıklı top_n sınıfı bul
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            
            results = []
            for idx in top_indices:
                if idx < len(self.rf_classifier.classes_):
                    destination_idx = self.rf_classifier.classes_[idx]
                    
                    # Destination adını bul
                    destination_name = None
                    for name, idx_val in self.destination_encodings.items():
                        if idx_val == destination_idx:
                            destination_name = name
                            break
                    
                    if destination_name is None:
                        destination_name = f"Destination_{destination_idx}"
                        logger.warning(f"Destinasyon adı bulunamadı, varsayılan değer kullanılıyor: {destination_name}")
                    
                    confidence = probabilities[idx]
                    
                    result = {
                        'destination': destination_name,
                        'confidence': float(confidence),
                        'reason': f"Bu destinasyon tercihlerinizle {confidence:.2f} oranında uyumlu (Random Forest)."
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Random Forest tahmin hatası: {str(e)}")
            return []
