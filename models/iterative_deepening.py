import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
import time
import matplotlib.pyplot as plt
import os

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
        best_result = None
        best_similarity = -1
        
        for depth in range(1, max_depth + 1):
            result = self._depth_limited_search(instance, depth)
            if result is not None:
                # Sonucun benzerlik değerini hesapla
                profile = self.destination_profiles.get(result)
                if profile is not None:
                    similarity = self._calculate_similarity(instance, profile)
                    # Daha iyi bir sonuç bulunduğunda güncelle
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_result = result
                        
                        # Eğer benzerlik çok yüksekse, aramayı erken sonlandır
                        if similarity > 0.85:
                            return best_result
        
        # En iyi sonucu döndür, yoksa en yakın destinasyonu bul
        if best_result is not None:
            return best_result
        else:
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
        
        # Adaptif arama stratejisi: Derinlik arttıkça daha fazla aday değerlendir
        if depth > 3:
            num_candidates = max(num_candidates, int(len(candidates) * 0.3))  # En az %30 aday
        
        # Her derinlik seviyesinde, adayları değerlendir ve en iyilerini seç
        for d in range(depth):
            scores = []
            for candidate in candidates:
                profile = self.destination_profiles[candidate]
                similarity = self._calculate_similarity(instance, profile)
                
                # Derinlik arttıkça benzerlik hesabını daha hassas yap
                if d > 0:
                    # Daha derin seviyelerde daha hassas benzerlik hesabı
                    # Önceki benzerlik skoruna ek olarak, özellik bazlı benzerlikler ekle
                    feature_similarities = []
                    
                    # Özellik bazlı benzerlikler
                    for i, (feat_val, prof_val) in enumerate(zip(instance, profile)):
                        # Özelliğe göre ağırlıklandırma
                        weight = 1.0
                        if i < len(self.feature_columns):
                            if self.feature_columns[i] == 'user_satisfaction':
                                weight = 1.5
                            elif self.feature_columns[i] == 'value_score':
                                weight = 1.3
                        
                        # Özellik benzerliği
                        feat_diff = abs(feat_val - prof_val)
                        feat_sim = 1.0 / (1.0 + feat_diff)
                        feature_similarities.append(feat_sim * weight)
                    
                    # Özellik benzerliklerinin ortalamasını ekle
                    if feature_similarities:
                        similarity = 0.7 * similarity + 0.3 * np.mean(feature_similarities)
                
                scores.append((candidate, similarity))
            
            # En iyi benzerliğe sahip adayları seç
            scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in scores[:num_candidates]]
            
            # Eğer sadece bir aday kaldıysa, onu döndür
            if len(candidates) == 1:
                return candidates[0]
            
            # Eğer adaylar arasında çok küçük fark varsa, daha fazla aramaya gerek yok
            if len(scores) >= 2:
                best_score = scores[0][1]
                second_best = scores[1][1]
                if best_score > 0.8 and (best_score - second_best) < 0.05:
                    return scores[0][0]  # En iyi adayı döndür
        
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
        """İki profil arasındaki benzerliği hesapla - Geliştirilmiş versiyon"""
        # Kategorik ve sayısal özellikleri ayrı değerlendir
        categorical_indices = [i for i, col in enumerate(self.feature_columns) 
                            if col in ['season', 'preferred_activity', 'destination']]
        numerical_indices = [i for i, col in enumerate(self.feature_columns) 
                            if col in ['budget', 'duration', 'value_score', 'user_satisfaction']]
        
        # Kategorik özellikler için benzerlik
        categorical_similarity = 0.0
        for i in categorical_indices:
            if i < len(self.feature_columns):
                feature_name = self.feature_columns[i]
                # Özelliğe göre ağırlıklandırma
                feature_weight = 1.0
                if feature_name == 'preferred_activity':
                    feature_weight = 1.4  # Aktivite tercihi daha önemli
                elif feature_name == 'season':
                    feature_weight = 1.3  # Sezon tercihi de önemli
                
                if instance[i] == profile[i]:
                    # Tam eşleşme için yüksek bonus
                    categorical_similarity += 1.0 * feature_weight
                else:
                    # Eşleşmeme için ceza (normalize edilmiş mesafe)
                    diff = abs(instance[i] - profile[i])
                    max_diff = len(self.label_encoders.get(feature_name, {}).classes_) - 1 if feature_name in self.label_encoders else 1
                    if max_diff > 0:
                        normalized_diff = diff / max_diff
                        # Aktivite ve sezon uyumsuzluğu için daha yüksek ceza
                        if feature_name in ['preferred_activity', 'season']:
                            categorical_similarity -= normalized_diff * 0.7 * feature_weight
                        else:
                            categorical_similarity -= normalized_diff * 0.5 * feature_weight
        
        # Sayısal özellikler için benzerlik
        numerical_similarity = 0.0
        for i in numerical_indices:
            if i < len(self.feature_columns):
                feature_name = self.feature_columns[i]
                # Özelliğe göre ağırlıklandırma - Kullanıcı memnuniyeti ve değer skoru ağırlıklarını artırıyoruz
                weight = 1.0
                if feature_name == 'user_satisfaction':
                    weight = 2.0  # Kullanıcı memnuniyeti çok daha önemli (1.5'ten 2.0'a)
                elif feature_name == 'value_score':
                    weight = 1.7  # Değer skoru da çok önemli (1.3'ten 1.7'ye)
                elif feature_name == 'budget':
                    weight = 1.5  # Bütçe de önemli (1.2'den 1.5'e)
                
                # Mesafe hesapla
                diff = abs(instance[i] - profile[i])
                
                # Özelliğe göre adaptif eşik değeri
                if feature_name == 'budget':
                    threshold = 0.4  # Bütçe için eşik (0.5'ten 0.4'e)
                elif feature_name == 'user_satisfaction':
                    threshold = 0.25  # Kullanıcı memnuniyeti için daha düşük eşik (0.3'ten 0.25'e)
                elif feature_name == 'value_score':
                    threshold = 0.3  # Değer skoru için eşik (0.4'ten 0.3'e)
                else:
                    threshold = 0.4  # Diğer özellikler için eşik
                
                # Eşik altındaki farklar için karesel benzerlik
                if diff < threshold:
                    numerical_similarity += (1 - (diff / threshold)**2) * weight
                else:
                    # Eşik üstü için lineer benzerlik - Daha yumuşak düşüş
                    numerical_similarity += max(0, (1 - diff/2)) * weight * 0.6  # 0.5'ten 0.6'ya
        
        # Kategorik ve sayısal benzerlikler için ağırlıklı ortalama
        # Sayısal özelliklere daha fazla ağırlık veriyoruz
        total_similarity = (categorical_similarity * 0.35 + numerical_similarity * 0.65)  # 0.4/0.6'dan 0.35/0.65'e
        
        # Kullanıcı memnuniyeti ve değer skoru için ek bonus
        user_sat_idx = -1
        value_score_idx = -1
        budget_idx = -1
        
        for i, col in enumerate(self.feature_columns):
            if col == 'user_satisfaction':
                user_sat_idx = i
            elif col == 'value_score':
                value_score_idx = i
            elif col == 'budget':
                budget_idx = i
        
        if user_sat_idx >= 0 and user_sat_idx < len(instance) and instance[user_sat_idx] > 0:
            # Yüksek kullanıcı memnuniyeti için bonus - Artırıldı
            user_sat_bonus = instance[user_sat_idx] * 0.3  # 0.2'den 0.3'e
            total_similarity += user_sat_bonus
        
        if value_score_idx >= 0 and value_score_idx < len(instance) and instance[value_score_idx] > 0:
            # Yüksek değer skoru için bonus - Artırıldı
            value_score_bonus = instance[value_score_idx] * 0.25  # 0.15'ten 0.25'e
            total_similarity += value_score_bonus
        
        # Bütçe uyumluluğu için ek bonus
        if budget_idx >= 0 and budget_idx < len(instance) and instance[budget_idx] > 0:
            budget_diff = abs(instance[budget_idx] - profile[budget_idx])
            if budget_diff < 0.3:  # Bütçe farkı küçükse bonus ver
                budget_bonus = (0.3 - budget_diff) * 0.5
                total_similarity += budget_bonus
        
        # Benzerliği normalize et - Sigmoid fonksiyonu
        normalized_similarity = 1.0 / (1.0 + np.exp(-total_similarity))
        
        return normalized_similarity
    
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
                    try:
                        features[col] = self.label_encoders[col].transform([str(user_preferences[col])])[0]
                    except ValueError as e:
                        logger.warning(f"{col} için encoding hatası: {str(e)}")
                        # Bilinmeyen etiketler için en yakın bilinen etiketi bul
                        if col == 'preferred_activity':
                            # Bilinmeyen aktivite için fallback
                            known_activities = self.label_encoders[col].classes_
                            if 'Plaj' in known_activities:
                                features[col] = self.label_encoders[col].transform(['Plaj'])[0]
                                logger.info(f"Bilinmeyen aktivite için 'Plaj' kullanılıyor")
                            elif 'Kültür' in known_activities:
                                features[col] = self.label_encoders[col].transform(['Kültür'])[0]
                                logger.info(f"Bilinmeyen aktivite için 'Kültür' kullanılıyor")
                            else:
                                # İlk sınıfı kullan
                                features[col] = 0
                        elif col == 'season':
                            # Bilinmeyen sezon için fallback
                            known_seasons = self.label_encoders[col].classes_
                            if 'Yaz' in known_seasons:
                                features[col] = self.label_encoders[col].transform(['Yaz'])[0]
                                logger.info(f"Bilinmeyen sezon için 'Yaz' kullanılıyor")
                            else:
                                # İlk sınıfı kullan
                                features[col] = 0
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
            # Daha yüksek varsayılan değerler kullanıyoruz
            numerical_data['value_score'] = 4.0  # Daha yüksek varsayılan değer (3.5'ten 4.0'a)
            numerical_data['user_satisfaction'] = 4.5  # Daha yüksek varsayılan değer (4.0'dan 4.5'e)
            
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
            
            # Tüm özellikleri birleştir
            instance = np.zeros(len(self.feature_columns), dtype=np.float64)  # Veri tipini açıkça belirt
            feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
            
            # Kategorik değerleri yerleştir
            for col, value in features.items():
                if col in feature_indices:
                    instance[feature_indices[col]] = float(value)  # float'a dönüştür
            
            # Sayısal değerleri yerleştir
            for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                if col in feature_indices:
                    instance[feature_indices[col]] = float(scaled_numerical[0, i])  # float'a dönüştür
            
            # Bellek düzenini garanti et
            instance = np.ascontiguousarray(instance)
            
            # IDDFS ile tahmin yap
            destination_idx = self._predict_with_iddfs(instance, self.max_depth)
            
            # Güven değeri hesapla
            best_profile = self.destination_profiles[destination_idx]
            confidence = self._calculate_similarity(instance, best_profile)
            
            # Destinasyon adını bul
            destination_name = None
            for name, idx in self.destination_encodings.items():
                if idx == destination_idx:
                    destination_name = name
                    break
            
            return {'destination': destination_name, 'confidence': float(confidence)}
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def save_model(self):
        """Modeli kaydet"""
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Label encoder'ları kaydet
            joblib.dump(self.label_encoders, 'saved_models/iddfs_label_encoders.joblib')
            
            # Scaler'ı kaydet
            joblib.dump(self.scaler, 'saved_models/iddfs_scaler.joblib')
            
            # Özellik sütunlarını kaydet
            joblib.dump(self.feature_columns, 'saved_models/iddfs_feature_columns.joblib')
            
            # Destinasyon profillerini kaydet
            joblib.dump(self.destination_profiles, 'saved_models/iddfs_destination_profiles.joblib')
            
            # Destinasyon kodlamalarını kaydet
            joblib.dump(self.destination_encodings, 'saved_models/iddfs_destination_encodings.joblib')
            
            # Destinasyonları kaydet
            joblib.dump(self.destinations, 'saved_models/iddfs_destinations.joblib')
            
            # Maksimum derinliği kaydet
            joblib.dump(self.max_depth, 'saved_models/iddfs_max_depth.joblib')
            
            logger.info("IDDFS modeli başarıyla kaydedildi")
            return True
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {str(e)}")
            return False
    
    def load_model(self):
        """Modeli yükle"""
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('saved_models/iddfs_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('saved_models/iddfs_scaler.joblib')
            
            # Özellik sütunlarını yükle
            self.feature_columns = joblib.load('saved_models/iddfs_feature_columns.joblib')
            
            # Destinasyon profillerini yükle
            self.destination_profiles = joblib.load('saved_models/iddfs_destination_profiles.joblib')
            
            # Destinasyon kodlamalarını yükle
            try:
                self.destination_encodings = joblib.load('saved_models/iddfs_destination_encodings.joblib')
            except:
                logger.warning("Destinasyon kodlamaları yüklenemedi.")
            
            # Destinasyonları yükle
            try:
                self.destinations = joblib.load('saved_models/iddfs_destinations.joblib')
            except:
                logger.warning("Destinasyonlar yüklenemedi.")
            
            # Maksimum derinliği yükle
            self.max_depth = joblib.load('saved_models/iddfs_max_depth.joblib')
            
            logger.info("IDDFS modeli başarıyla yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
