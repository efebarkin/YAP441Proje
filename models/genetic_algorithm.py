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
        
        # Sayısal değişkenleri ölçeklendir
        numerical_features = ['budget', 'duration', 'value_score', 'user_satisfaction']
        self.scaler = StandardScaler()
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Feature'ları ve target'ı ayır
        self.feature_columns = ['season', 'preferred_activity', 'destination', 'budget', 'duration', 'value_score', 'user_satisfaction']
        X = df[self.feature_columns].values
        
        # Hedef değişkeni sayısal hale getir
        le_target = LabelEncoder()
        y = le_target.fit_transform(df['recommended_vacation'].astype(str))
        self.label_encoders['recommended_vacation'] = le_target
        
        # Benzersiz hedef değerlerini al
        unique_destinations = np.unique(y)
        self.num_classes = len(unique_destinations)
        
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
        
        for generation in range(generations):
            # Her bireyin uygunluğunu hesapla
            fitness_scores = []
            for individual in population:
                fitness = self._calculate_fitness(individual, X_train, y_train)
                fitness_scores.append(fitness)
            
            # En iyi bireyi bul ve kaydet
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            self.best_individual = population[best_idx]
            self.fitness_history.append(best_fitness)
            
            # Log ilerleme
            if (generation + 1) % 5 == 0 or generation == 0:
                logger.info(f"Jenerasyon {generation + 1}/{generations}, En İyi Uygunluk: {best_fitness:.4f}")
            
            # Erken durdurma kontrolü
            if abs(best_fitness - last_best_fitness) < 0.0001:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            last_best_fitness = best_fitness
            
            # 10 jenerasyon boyunca ilerleme yoksa dur
            if stagnation_count >= 10 and best_fitness > 0.05:  # Minimum eşik değeri artırıldı
                logger.info(f"Erken durdurma: 10 jenerasyon boyunca ilerleme yok. Jenerasyon {generation + 1}/{generations}")
                break
            
            # Adaptif mutasyon oranı: Popülasyon çeşitliliği azaldıkça mutasyon oranını artır
            population_diversity = np.std([np.mean(ind) for ind in population])
            adaptive_mutation_rate = mutation_rate * (1.0 + (0.5 / (population_diversity + 0.1)))
            
            # Yeni popülasyon oluştur
            new_population = []
            
            # Seçkincilik: En iyi bireyi direkt olarak bir sonraki jenerasyona aktar
            new_population.append(self.best_individual)
            
            # Geriye kalan bireyleri turnuva seçimi, çaprazlama ve mutasyon ile oluştur
            while len(new_population) < population_size:
                # Turnuva seçimi
                parent1 = self._tournament_selection(population, fitness_scores, tournament_size)
                parent2 = self._tournament_selection(population, fitness_scores, tournament_size)
                
                # Çaprazlama
                child = self._crossover(parent1, parent2)
                
                # Mutasyon
                if random.random() < adaptive_mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            # Yeni popülasyonu güncelle
            population = new_population
        
        # En iyi bireyi model olarak kaydet
        self.feature_weights = self.best_individual
        
        # Test seti üzerinde değerlendir
        y_pred = self.predict_batch(X_test)
        
        try:
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"\nTest Seti Doğruluk Oranı: {accuracy:.4f}")
            
            logger.info("\nConfusion Matrix:")
            logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
            
            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
        except Exception as e:
            logger.error(f"Değerlendirme hatası: {str(e)}")
        
        # Grafik oluştur: Uygunluk Değeri vs Jenerasyon
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, marker='o')
            plt.title('Genetik Algoritma Eğitim İlerlemesi')
            plt.xlabel('Jenerasyon')
            plt.ylabel('En İyi Uygunluk Değeri')
            plt.grid(True)
            plt.savefig('models/genetic_algorithm_fitness.png', dpi=300, bbox_inches='tight')
            logger.info("Genetik algoritma eğitim grafiği kaydedildi: models/genetic_algorithm_fitness.png")
        except Exception as e:
            logger.warning(f"Grafik oluşturma hatası: {str(e)}")
        
        logger.info("Genetik Algoritma model eğitimi tamamlandı")

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
    
    def _mutate(self, individual):
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
    
    def predict_batch(self, X):
        """Toplu tahmin yap"""
        if self.feature_weights is None:
            logger.error("Model eğitilmemiş!")
            return None
        
        y_pred = []
        
        for instance in X:
            # Her sınıf için bir skor hesapla
            scores = np.zeros(self.num_classes)
            for i, feature_val in enumerate(instance):
                for j in range(self.num_classes):
                    # Özellik değeri ve ağırlığın çarpımını skora ekle
                    scores[j] += feature_val * self.feature_weights[i, j]
            
            # En yüksek skora sahip sınıfı seç
            predicted_class = np.argmax(scores)
            y_pred.append(predicted_class)
        
        return np.array(y_pred)
    
    def predict(self, user_preferences):
        """Kullanıcı tercihlerine göre tatil önerisi yap"""
        logger.info("Genetik Algoritma ile tahmin yapılıyor...")
        
        try:
            # Kullanıcı tercihlerini işle
            processed_preferences = self._process_user_preferences(user_preferences)
            
            # Özellikleri çıkar
            features = []
            for col in self.feature_columns:
                if col in processed_preferences:
                    features.append(processed_preferences[col])
                else:
                    # Eğer özellik yoksa, varsayılan değer kullan
                    features.append(0)
            
            features = np.array(features).reshape(1, -1)
            
            # Her sınıf için bir skor hesapla
            scores = np.zeros(self.num_classes)
            for i, feature_val in enumerate(features[0]):
                for j in range(self.num_classes):
                    scores[j] += feature_val * self.feature_weights[i, j]
            
            # En yüksek skora sahip sınıfı seç
            predicted_class = np.argmax(scores)
            
            # Sınıf etiketini çöz
            if 'recommended_vacation' in self.label_encoders:
                predicted_destination = self.label_encoders['recommended_vacation'].inverse_transform([predicted_class])[0]
            else:
                predicted_destination = f"Destination_{predicted_class}"
            
            # Normalize edilmiş skorları hesapla (güven değerleri olarak)
            confidence_scores = softmax(scores)
            algorithm_confidence = confidence_scores[predicted_class]
            
            # Sonucu döndür
            result = {
                "destination": predicted_destination,
                "season": user_preferences.get("season", ""),
                "activity": user_preferences.get("preferred_activity", ""),
                "costs": {
                    "hotel_price": round(random.uniform(500, 3000)),
                    "flight_cost": round(random.uniform(1000, 5000)),
                    "total_cost": 0  # Sonra hesaplanacak
                },
                "algorithm_confidence": float(algorithm_confidence)
            }
            
            # Toplam maliyeti hesapla
            result["costs"]["total_cost"] = result["costs"]["hotel_price"] * user_preferences.get("duration", 7) + result["costs"]["flight_cost"]
            
            logger.info(f"Genetik Algoritma tahmini: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Genetik Algoritma tahmin hatası: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _process_user_preferences(self, user_preferences):
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
        
        return instance
    
    def save_model(self):
        """Modeli kaydet"""
        if self.feature_weights is None:
            logger.error("Kaydedilecek model yok!")
            return
        
        # Modeli kaydet
        joblib.dump(self.feature_weights, 'models/genetic_model.joblib')
        
        # Label encoder'ları kaydet
        joblib.dump(self.label_encoders, 'models/genetic_label_encoders.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, 'models/genetic_scaler.joblib')
        
        # Feature kolonlarını kaydet
        joblib.dump(self.feature_columns, 'models/genetic_feature_columns.joblib')
        
        # Sınıf sayısını kaydet
        joblib.dump(self.num_classes, 'models/genetic_num_classes.joblib')
        
        logger.info("Genetik Algoritma modeli kaydedildi")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            # Modeli yükle
            self.feature_weights = joblib.load('models/genetic_model.joblib')
            
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('models/genetic_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('models/genetic_scaler.joblib')
            
            # Feature kolonlarını yükle
            self.feature_columns = joblib.load('models/genetic_feature_columns.joblib')
            
            # Sınıf sayısını yükle
            self.num_classes = joblib.load('models/genetic_num_classes.joblib')
            
            logger.info("Genetik Algoritma modeli yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
