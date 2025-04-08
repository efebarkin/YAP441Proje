import pandas as pd
import numpy as np
import random
import logging
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.special import softmax
import traceback

logger = logging.getLogger(__name__)

class GeneticVacationRecommender:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.destinations = []
        self.feature_weights = None
        self.best_individual = None
        self.fitness_history = []
        
    def train(self, df, population_size=100, generations=50, mutation_rate=0.1, tournament_size=5):
        logger.info("Genetik Algoritma model eğitimi başlıyor...")
        
        # Gereksiz sütunları kaldır
        columns_to_drop = ['user_id', 'hotel_price_per_night', 'flight_cost', 'total_cost']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
        
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
        
        # Sayısal değişkenleri ölçeklendir
        numerical_features = ['budget', 'duration', 'value_score', 'user_satisfaction']
        self.scaler = StandardScaler()
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Feature'ları ve target'ı ayır
        self.feature_columns = ['season', 'preferred_activity', 'budget', 'duration', 'value_score', 'user_satisfaction']
        X = df[self.feature_columns].values
        
        # Hedef değişkeni sayısal hale getir
        y = df['destination'].values
        self.num_classes = len(np.unique(y))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Genetik Algoritma: Başlangıç popülasyonu oluştur
        # Her birey, her özellik için bir ağırlık içerir (özellik boyutu x sınıf sayısı kadar)
        population = []
        for _ in range(population_size):
            # Her özellik için her sınıfa bir ağırlık ata
            # Daha iyi bir başlangıç: Küçük değerlerle başla
            individual = np.random.uniform(-0.5, 0.5, (len(self.feature_columns), self.num_classes))
            population.append(individual)
        
        # Daha iyi bir başlangıç: Veri tabanlı bir birey ekle
        data_based_individual = np.zeros((len(self.feature_columns), self.num_classes))
        for j in range(self.num_classes):
            # Bu sınıfa ait örnekleri seç
            class_samples = X_train[y_train == j]
            if len(class_samples) > 0:
                # Her özellik için ortalama değeri hesapla
                feature_means = np.mean(class_samples, axis=0)
                data_based_individual[:, j] = feature_means
        
        # Veri tabanlı bireyi normalize et
        data_based_individual = data_based_individual / (np.max(np.abs(data_based_individual)) + 1e-10)
        population.append(data_based_individual)
        
        # Genetik Algoritma: Jenerasyonlar boyunca evolve et
        stagnation_count = 0
        last_best_fitness = 0
        best_fitness_overall = 0
        early_stopping_patience = 10  # Erken durdurma için sabır parametresi
        
        for generation in range(generations):
            # Her bireyin uygunluğunu hesapla
            fitness_scores = []
            for individual in population:
                fitness = self._calculate_fitness(individual, X_train, y_train)
                fitness_scores.append(fitness)
            
            # En iyi bireyi bul
            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            # Fitness geçmişini kaydet
            self.fitness_history.append(best_fitness)
            
            # En iyi bireyi sakla
            if best_fitness > best_fitness_overall:
                self.best_individual = best_individual.copy()
                best_fitness_overall = best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Her 10 jenerasyonda bir ilerleme raporu
            if generation % 10 == 0 or generation == generations - 1:
                logger.info(f"Jenerasyon {generation+1}/{generations}, En İyi Uygunluk: {best_fitness:.4f}")
            
            # Erken durdurma kontrolü
            if stagnation_count >= early_stopping_patience:
                logger.info(f"Jenerasyon {generation+1}/{generations} - Erken durdurma (En iyi uygunluk: {best_fitness_overall:.4f})")
                break
            
            # Yeni popülasyon oluştur
            new_population = []
            
            # Elitizm: En iyi bireyi doğrudan yeni popülasyona aktar
            new_population.append(self.best_individual.copy())
            
            # Yeni popülasyonu doldur
            while len(new_population) < population_size:
                # Turnuva seçimi ile ebeveynleri seç
                parent1 = self._tournament_selection(population, fitness_scores, tournament_size)
                parent2 = self._tournament_selection(population, fitness_scores, tournament_size)
                
                # Çaprazlama
                if random.random() < 0.8:  # Çaprazlama olasılığı
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutasyon
                child = self._mutate(child, mutation_rate)
                
                new_population.append(child)
            
            # Yeni popülasyonu güncelle
            population = new_population
        
        # Final modeli değerlendir
        self.model = self.best_individual
        self.feature_weights = self.best_individual  # Ensure both variables point to the same data
        
        # Test seti üzerinde değerlendir
        y_pred = self._predict_batch(X_test, self.model)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Seti Doğruluk: {accuracy:.4f}")
        
        # Confusion matrix ve classification report
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Fitness geçmişini görselleştir
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Genetik Algoritma Eğitim İlerlemesi')
        plt.xlabel('Jenerasyon')
        plt.ylabel('En İyi Uygunluk')
        plt.grid(True)
        plt.savefig('models/genetic_fitness_history.png')
        
        logger.info("Genetik Algoritma model eğitimi tamamlandı")
        return self

    def _calculate_fitness(self, individual, X, y):
        y_pred = []
        
        for instance in X:
            # Her sınıf için bir skor hesapla
            scores = np.zeros(self.num_classes)
            for i, feature_val in enumerate(instance):
                for j in range(self.num_classes):
                    # Özellik değeri ve ağırlığın çarpımını skora ekle
                    scores[j] += feature_val * individual[i, j]
            
            # En yüksek skora sahip sınıfı seç
            predicted_class = np.argmax(scores)
            y_pred.append(predicted_class)
        
        # Doğruluk oranını uygunluk değeri olarak kullan
        fitness = accuracy_score(y, y_pred)
        
        # Ağırlıkların çok büyük olmasını cezalandır (L2 regularizasyon)
        l2_penalty = 0.01 * np.sum(individual**2)
        
        # Ağırlıkların çeşitliliğini teşvik et
        diversity_bonus = 0.05 * np.std(individual)
        
        # Nihai uygunluk değeri
        final_fitness = fitness - l2_penalty + diversity_bonus
        
        # Eğer fitness çok düşükse, minimum bir değer döndür
        return max(0.0001, final_fitness)  # Minimum pozitif değer döndür

    def _tournament_selection(self, population, fitness_scores, tournament_size):
        """Turnuva seçimi ile ebeveyn seç"""
        # Rastgele bireyler seç
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # En iyi uygunluk değerine sahip bireyi döndür
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1, parent2):
        """İki ebeveynden yeni birey oluştur (çaprazlama)"""
        # Her özellik için rastgele ebeveynden gen al
        child = np.zeros_like(parent1)
        for i in range(parent1.shape[0]):
            for j in range(parent1.shape[1]):
                if random.random() < 0.5:
                    child[i, j] = parent1[i, j]
                else:
                    child[i, j] = parent2[i, j]
        return child

    def _mutate(self, individual, mutation_rate):
        """Bireyde rastgele mutasyonlar gerçekleştir"""
        mutated = individual.copy()
        
        # Daha akıllı mutasyon: Tüm genleri değil, sadece bazılarını değiştir
        mutation_count = random.randint(1, int(mutated.size * 0.1) + 1)  # Maksimum %10 gen mutasyona uğrasın
        
        for _ in range(mutation_count):
            # Rastgele bir pozisyon seç
            i = random.randint(0, mutated.shape[0] - 1)
            j = random.randint(0, mutated.shape[1] - 1)
            
            # Mevcut değere küçük bir değişiklik ekle (tamamen rastgele değil)
            current_val = mutated[i, j]
            mutated[i, j] = current_val + random.uniform(-0.3, 0.3)
            
            # Değeri makul bir aralıkta tut
            mutated[i, j] = max(-1.0, min(1.0, mutated[i, j]))
        
        return mutated

    def _predict_batch(self, X, model):
        """Toplu tahmin yap"""
        y_pred = []
        
        for instance in X:
            # Her sınıf için bir skor hesapla
            scores = np.zeros(self.num_classes)
            for i, feature_val in enumerate(instance):
                for j in range(self.num_classes):
                    # Özellik değeri ve ağırlığın çarpımını skora ekle
                    scores[j] += feature_val * model[i, j]
            
            # En yüksek skora sahip sınıfı seç
            predicted_class = np.argmax(scores)
            y_pred.append(predicted_class)
        
        return np.array(y_pred)

    def predict(self, user_preferences, top_n=5, population_size=None, generations=None):
        """Kullanıcı tercihlerine göre destinasyon tahmini yap
        
        Args:
            user_preferences (dict): Kullanıcı tercihleri
            top_n (int): Döndürülecek öneri sayısı
            population_size (int, optional): Genetik algoritma için popülasyon boyutu
            generations (int, optional): Genetik algoritma için jenerasyon sayısı
        """
        try:
            # Kategorik değişkenler için özellikler
            features = {}
            
            # Sayısal değişkenler için veri
            numerical_data = {}
            
            # Kategorik değişkenleri işle
            if 'season' in user_preferences:
                season = str(user_preferences['season'])
                if 'season' in self.label_encoders and season in self.label_encoders['season'].classes_:
                    features['season'] = self.label_encoders['season'].transform([season])[0]
                else:
                    logger.warning(f"Bilinmeyen sezon: {season}")
                    features['season'] = 0  # Varsayılan değer
            
            if 'preferred_activity' in user_preferences:
                activity = str(user_preferences['preferred_activity'])
                if 'preferred_activity' in self.label_encoders and activity in self.label_encoders['preferred_activity'].classes_:
                    features['preferred_activity'] = self.label_encoders['preferred_activity'].transform([activity])[0]
                else:
                    logger.warning(f"Bilinmeyen aktivite: {activity}")
                    features['preferred_activity'] = 0  # Varsayılan değer
            
            # Sayısal değişkenleri işle
            for col in ['budget', 'duration']:
                if col in user_preferences:
                    numerical_data[col] = float(user_preferences[col])
                else:
                    # Varsayılan değerler
                    numerical_data[col] = 5000.0 if col == 'budget' else 7.0
            
            # Olmayan değerleri tahmin et
            numerical_data['value_score'] = 3.5  # Ortalama bir değer
            numerical_data['user_satisfaction'] = 4.0  # Ortalama bir değer
            
            # Sayısal değişkenleri ölçeklendir
            numerical_features = pd.DataFrame({
                'budget': [numerical_data['budget']],
                'duration': [numerical_data['duration']],
                'value_score': [numerical_data['value_score']],
                'user_satisfaction': [numerical_data['user_satisfaction']]
            })
            
            scaled_numerical = self.scaler.transform(numerical_features)
            
            # Özellik vektörünü oluştur
            instance = np.zeros(len(self.feature_columns))
            feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
            
            # Kategorik değişkenleri ekle
            for col, value in features.items():
                if col in feature_indices:
                    instance[feature_indices[col]] = value
            
            # Sayısal değişkenleri ekle
            for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
                if col in feature_indices:
                    instance[feature_indices[col]] = scaled_numerical[0, i]
            
            # Tahmin yap
            scores = np.dot(instance, self.model)
            
            # Min-max normalizasyonu ile skorları 0.6-1.0 arasına ölçeklendir
            # Bu, daha yüksek güven skorları üretecek
            if len(scores) > 1:
                min_score = np.min(scores)
                max_score = np.max(scores)
                if max_score > min_score:  # Bölme hatası olmaması için kontrol
                    normalized_scores = 0.6 + 0.4 * (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(scores) * 0.7  # Tüm skorlar eşitse
            else:
                normalized_scores = np.array([0.8])  # Tek bir skor varsa
            
            # Softmax ile olasılık dağılımına çevir, ancak daha keskin bir dağılım için sıcaklık parametresi ekle
            temperature = 0.5  # Daha düşük sıcaklık, daha keskin bir dağılım üretir
            probabilities = softmax(scores / temperature)
            
            # Önce tüm destinasyonları sırala
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # Eğer destinasyon sayısı top_n'den azsa, her destinasyon için birden fazla öneri oluştur
            if len(sorted_indices) < top_n and len(sorted_indices) > 0:
                # Her destinasyon için kaç öneri oluşturulacağını hesapla
                repeats = max(1, top_n // len(sorted_indices))
                # Tüm destinasyonlar için tekrarlı öneriler oluştur
                expanded_indices = []
                for idx in sorted_indices:
                    for _ in range(repeats):
                        expanded_indices.append(idx)
                # En fazla top_n kadar öneri al
                top_indices = expanded_indices[:top_n]
            else:
                # Normal durumda en yüksek olasılıklı top_n destinasyonu al
                top_indices = sorted_indices[:top_n]
            
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
            
            results = []
            for idx in top_indices:
                # Destinations listesi kontrolü
                if not self.destinations or idx >= len(self.destinations):
                    # Eğer destinations listesi yoksa veya index geçersizse, label encoder'dan destinasyonları al
                    if 'destination' in self.label_encoders:
                        self.destinations = list(self.label_encoders['destination'].classes_)
                    else:
                        # Hiçbir şekilde destinasyon bilgisi yoksa, index'i kullan
                        destination = f"Destination_{idx}"
                        logger.warning(f"Destinasyon bilgisi bulunamadı, varsayılan değer kullanılıyor: {destination}")
                
                # Destinasyon bilgisini al
                if self.destinations and idx < len(self.destinations):
                    destination = self.destinations[idx]
                else:
                    destination = f"Destination_{idx}"
                    
                # Hem normalize edilmiş skoru hem de olasılığı kullanarak daha anlamlı bir güven skoru hesapla
                raw_confidence = float(probabilities[idx])
                norm_confidence = float(normalized_scores[idx])
                
                # İki skoru birleştirerek daha dengeli bir güven skoru oluştur
                # Yüksek olasılık ve yüksek normalize skor = yüksek güven
                # Aynı destinasyon için farklı öneriler oluşturmak için küçük varyasyonlar ekle
                variation = np.random.uniform(-0.05, 0.05)  # +-5% varyasyon
                base_confidence = min(1.0, max(0.6, (raw_confidence + norm_confidence) / 2))
                confidence = min(1.0, max(0.6, base_confidence + variation))
                
                # Güven skorunu yüzde olarak formatla
                confidence_percent = confidence * 100
                
                # Destinasyona uygun mevsim ve aktivite seç
                suitable_seasons = [s for s, d in season_destinations.items() if destination in d]
                suitable_activities = [a for a, d in activity_destinations.items() if destination in d]
                
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
                
                # Farklı açıklamalar oluştur
                reasons = [
                    f"Bu destinasyon {season} mevsiminde %{confidence_percent:.1f} oranında tercihlerinize uygun.",
                    f"{destination}, {activity} aktivitesi için %{confidence_percent:.1f} oranında uyumlu.",
                    f"{destination} bütçenize ve sürenize %{confidence_percent:.1f} oranında uygun.",
                    f"Tercihlerinize göre {destination} %{confidence_percent:.1f} oranında iyi bir seçim.",
                    f"{season} mevsiminde {destination} %{confidence_percent:.1f} oranında keyifli bir tatil sunabilir."
                ]
                reason = reasons[np.random.randint(0, len(reasons))]
                
                result = {
                    'destination': destination,
                    'confidence': confidence,
                    'reason': reason,
                    'season': season,
                    'preferred_activity': activity
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def save_model(self):
        """Modeli kaydet"""
        if self.model is None:
            logger.error("Model eğitilmemiş! Kaydedilecek model yok.")
            return
        
        # Model parametrelerini kaydet
        joblib.dump(self.model, 'models/genetic_model.joblib')
        joblib.dump(self.label_encoders, 'models/genetic_label_encoders.joblib')
        joblib.dump(self.scaler, 'models/genetic_scaler.joblib')
        joblib.dump(self.feature_columns, 'models/genetic_feature_columns.joblib')
        joblib.dump(self.destinations, 'models/genetic_destinations.joblib')
        joblib.dump(self.num_classes, 'models/genetic_num_classes.joblib')
        
        logger.info("Genetik Algoritma modeli kaydedildi")

    def load_model(self):
        """Modeli yükle"""
        try:
            # Modeli yükle
            self.model = joblib.load('models/genetic_model.joblib')
            
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('models/genetic_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('models/genetic_scaler.joblib')
            
            # Feature kolonlarını yükle
            self.feature_columns = joblib.load('models/genetic_feature_columns.joblib')
            
            # Destinasyonları yükle
            self.destinations = joblib.load('models/genetic_destinations.joblib')
            
            # Sınıf sayısını yükle
            self.num_classes = joblib.load('models/genetic_num_classes.joblib')
            
            logger.info("Genetik Algoritma modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
