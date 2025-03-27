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
        
        # Kullanıcı memnuniyeti ve değer skoru için ek ağırlıklandırma
        # Kullanıcı memnuniyeti indeksini bul
        user_sat_idx = -1
        value_score_idx = -1
        for i, col in enumerate(self.feature_columns):
            if col == 'user_satisfaction':
                user_sat_idx = i
            elif col == 'value_score':
                value_score_idx = i
        
        # Kullanıcı memnuniyeti ve değer skoru için ek bonus
        user_sat_bonus = 0
        value_score_bonus = 0
        
        if user_sat_idx >= 0:
            # Kullanıcı memnuniyeti için ağırlıkların ortalaması
            user_sat_weights = np.mean(individual[user_sat_idx])
            # Kullanıcı memnuniyeti için bonus - pozitif ağırlıkları ödüllendir
            user_sat_bonus = max(0, user_sat_weights) * 0.25
            
            # Kullanıcı memnuniyeti ağırlıklarının varyansını kontrol et
            # Daha tutarlı ağırlıklar için ek bonus
            user_sat_variance = np.var(individual[user_sat_idx])
            if user_sat_variance < 0.1:  # Düşük varyans = tutarlı ağırlıklar
                user_sat_bonus += 0.05
        
        if value_score_idx >= 0:
            # Değer skoru için ağırlıkların ortalaması
            value_score_weights = np.mean(individual[value_score_idx])
            # Değer skoru için bonus - pozitif ağırlıkları ödüllendir
            value_score_bonus = max(0, value_score_weights) * 0.15
        
        # Ağırlıkların çok büyük olmasını cezalandır (L2 regularizasyon)
        l2_penalty = 0.01 * np.sum(individual**2)
        
        # Ağırlıkların çeşitliliğini teşvik et
        diversity_bonus = 0.05 * np.std(individual)
        
        # Nihai uygunluk değeri - kullanıcı memnuniyeti ve değer skoru için bonus ekle
        final_fitness = fitness + user_sat_bonus + value_score_bonus - l2_penalty + diversity_bonus
        
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
        # Çaprazlama tipi seçimi - adaptive çaprazlama tipini ekledik
        crossover_type = random.choice(["uniform", "single_point", "two_point", "blend", "adaptive"])
        
        child = np.zeros_like(parent1)
        
        if crossover_type == "uniform":
            # Uniform çaprazlama: Her özellik için rastgele ebeveynden gen al
            for i in range(parent1.shape[0]):
                for j in range(parent1.shape[1]):
                    if random.random() < 0.5:
                        child[i, j] = parent1[i, j]
                    else:
                        child[i, j] = parent2[i, j]
        
        elif crossover_type == "single_point":
            # Tek noktalı çaprazlama: Rastgele bir noktadan sonraki tüm genler diğer ebeveynden gelir
            # Her satır için ayrı çaprazlama noktası
            for i in range(parent1.shape[0]):
                crossover_point = random.randint(0, parent1.shape[1] - 1)
                child[i, :crossover_point] = parent1[i, :crossover_point]
                child[i, crossover_point:] = parent2[i, crossover_point:]
        
        elif crossover_type == "two_point":
            # İki noktalı çaprazlama: İki nokta arasındaki genler bir ebeveynden, diğerleri diğer ebeveynden gelir
            # Her satır için ayrı çaprazlama noktaları
            for i in range(parent1.shape[0]):
                point1 = random.randint(0, parent1.shape[1] - 2)
                point2 = random.randint(point1 + 1, parent1.shape[1] - 1)
                
                child[i, :point1] = parent1[i, :point1]
                child[i, point1:point2] = parent2[i, point1:point2]
                child[i, point2:] = parent1[i, point2:]
        
        elif crossover_type == "blend":
            # Blend çaprazlama: İki ebeveynin ağırlıklı ortalaması
            # Her gen için farklı karışım oranı
            for i in range(parent1.shape[0]):
                for j in range(parent1.shape[1]):
                    # Rastgele bir karışım oranı seç
                    alpha = random.random()
                    # İki ebeveynin ağırlıklı ortalaması
                    child[i, j] = alpha * parent1[i, j] + (1 - alpha) * parent2[i, j]
        
        elif crossover_type == "adaptive":
            # Adaptive çaprazlama: Ebeveynlerin fitness değerlerine göre ağırlıklı çaprazlama
            # Bu yeni bir çaprazlama tipi olarak eklendi
            # Ebeveynlerin fitness değerlerini tahmin etmek için basit bir yöntem kullanıyoruz
            parent1_fitness_estimate = np.mean(np.abs(parent1))
            parent2_fitness_estimate = np.mean(np.abs(parent2))
            
            # Toplam fitness
            total_fitness = parent1_fitness_estimate + parent2_fitness_estimate
            
            if total_fitness > 0:
                # Ebeveynlerin fitness oranları
                parent1_ratio = parent1_fitness_estimate / total_fitness
                parent2_ratio = parent2_fitness_estimate / total_fitness
                
                # Her gen için ağırlıklı çaprazlama
                for i in range(parent1.shape[0]):
                    for j in range(parent1.shape[1]):
                        # Fitness değerlerine göre ağırlıklı ortalama
                        child[i, j] = parent1_ratio * parent1[i, j] + parent2_ratio * parent2[i, j]
                        
                        # Küçük bir rastgele varyasyon ekle
                        child[i, j] += random.uniform(-0.05, 0.05)
            else:
                # Eğer fitness tahminleri sıfırsa, uniform çaprazlama kullan
                for i in range(parent1.shape[0]):
                    for j in range(parent1.shape[1]):
                        if random.random() < 0.5:
                            child[i, j] = parent1[i, j]
                        else:
                            child[i, j] = parent2[i, j]
        
        # Kullanıcı memnuniyeti ve değer skoru için özel işlem
        # Bu özelliklerin ağırlıklarını korumak veya artırmak için
        user_sat_idx = -1
        value_score_idx = -1
        
        for i, col in enumerate(self.feature_columns):
            if col == 'user_satisfaction':
                user_sat_idx = i
            elif col == 'value_score':
                value_score_idx = i
        
        # Kullanıcı memnuniyeti için daha iyi olan ebeveynin genlerini tercih et
        if user_sat_idx >= 0:
            parent1_user_sat_mean = np.mean(parent1[user_sat_idx])
            parent2_user_sat_mean = np.mean(parent2[user_sat_idx])
            
            # Daha yüksek kullanıcı memnuniyeti ağırlığına sahip ebeveynden al
            if parent1_user_sat_mean > parent2_user_sat_mean:
                # Parent1'in kullanıcı memnuniyeti ağırlıkları daha iyi
                # Rastgele bir bonus ekle - bonus değerini artırıyoruz
                boost_factor = 1.0 + random.uniform(0, 0.3)  # %0-30 arası bonus (önceki %0-20)
                child[user_sat_idx] = parent1[user_sat_idx] * boost_factor
            else:
                # Parent2'nin kullanıcı memnuniyeti ağırlıkları daha iyi
                boost_factor = 1.0 + random.uniform(0, 0.3)  # %0-30 arası bonus (önceki %0-20)
                child[user_sat_idx] = parent2[user_sat_idx] * boost_factor
        
        # Değer skoru için benzer işlem
        if value_score_idx >= 0:
            parent1_value_mean = np.mean(parent1[value_score_idx])
            parent2_value_mean = np.mean(parent2[value_score_idx])
            
            if parent1_value_mean > parent2_value_mean:
                boost_factor = 1.0 + random.uniform(0, 0.2)  # %0-20 arası bonus (önceki %0-15)
                child[value_score_idx] = parent1[value_score_idx] * boost_factor
            else:
                boost_factor = 1.0 + random.uniform(0, 0.2)  # %0-20 arası bonus (önceki %0-15)
                child[value_score_idx] = parent2[value_score_idx] * boost_factor
        
        return child
    
    def _mutate(self, individual):
        """Bireyde rastgele mutasyonlar gerçekleştir"""
        mutated = individual.copy()
        
        # Daha akıllı mutasyon: Tüm genleri değil, sadece bazılarını değiştir
        mutation_count = random.randint(1, int(mutated.size * 0.1) + 1)  # Maksimum %10 gen mutasyona uğrasın
        
        # Kullanıcı memnuniyeti ve değer skoru indekslerini bul
        user_sat_idx = -1
        value_score_idx = -1
        
        for i, col in enumerate(self.feature_columns):
            if col == 'user_satisfaction':
                user_sat_idx = i
            elif col == 'value_score':
                value_score_idx = i
        
        # Özel mutasyon: Kullanıcı memnuniyeti ve değer skoru için pozitif yönde mutasyon
        if user_sat_idx >= 0 and random.random() < 0.3:  # %30 olasılıkla
            # Rastgele bir pozisyon seç
            j = random.randint(0, mutated.shape[1] - 1)
            
            # Mevcut değere pozitif bir değişiklik ekle
            current_val = mutated[user_sat_idx, j]
            # Pozitif yönde mutasyon - sadece artış
            mutated[user_sat_idx, j] = current_val + random.uniform(0, 0.3)
            
            # Değeri makul bir aralıkta tut
            mutated[user_sat_idx, j] = max(0.0, min(1.5, mutated[user_sat_idx, j]))
        
        if value_score_idx >= 0 and random.random() < 0.2:  # %20 olasılıkla
            # Rastgele bir pozisyon seç
            j = random.randint(0, mutated.shape[1] - 1)
            
            # Mevcut değere pozitif bir değişiklik ekle
            current_val = mutated[value_score_idx, j]
            # Pozitif yönde mutasyon - sadece artış
            mutated[value_score_idx, j] = current_val + random.uniform(0, 0.25)
            
            # Değeri makul bir aralıkta tut
            mutated[value_score_idx, j] = max(0.0, min(1.5, mutated[value_score_idx, j]))
        
        # Diğer genler için normal mutasyon
        for _ in range(mutation_count):
            # Rastgele bir pozisyon seç
            i = random.randint(0, mutated.shape[0] - 1)
            j = random.randint(0, mutated.shape[1] - 1)
            
            # Kullanıcı memnuniyeti ve değer skoru dışındaki genler için
            if i != user_sat_idx and i != value_score_idx:
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
            
            if processed_preferences is None:
                logger.error("Kullanıcı tercihleri işlenemedi.")
                return None
            
            # Özellikleri çıkar
            features = []
            for col in self.feature_columns:
                if isinstance(processed_preferences, dict) and col in processed_preferences:
                    features.append(processed_preferences[col])
                elif isinstance(processed_preferences, np.ndarray):
                    # Eğer processed_preferences bir numpy dizisi ise, doğrudan kullan
                    features = processed_preferences
                    break
                else:
                    # Eğer özellik yoksa, varsayılan değer kullan
                    features.append(0)
            
            # Numpy dizisine dönüştür ve veri tipini belirt
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float64).reshape(1, -1)
            else:
                features = features.reshape(1, -1).astype(np.float64)
            
            # Bellek düzenini garanti et
            features = np.ascontiguousarray(features)
            
            # Her sınıf için bir skor hesapla
            scores = np.zeros(self.num_classes)
            for i, feature_val in enumerate(features[0]):
                for j in range(self.num_classes):
                    # Kullanıcı memnuniyeti ve değer skoru için ek ağırlık
                    weight_multiplier = 1.0
                    if i < len(self.feature_columns):
                        col_name = self.feature_columns[i]
                        if col_name == 'user_satisfaction':
                            weight_multiplier = 1.5  # Kullanıcı memnuniyeti için 1.5x ağırlık
                        elif col_name == 'value_score':
                            weight_multiplier = 1.3  # Değer skoru için 1.3x ağırlık
                    
                    scores[j] += feature_val * self.feature_weights[i, j] * weight_multiplier
            
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
                "confidence": float(algorithm_confidence)
            }
            
            logger.info(f"Genetik Algoritma tahmini: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Genetik Algoritma tahmin hatası: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Hata durumunda fallback mekanizması
            try:
                # Eğer label_encoders varsa, rastgele bir destinasyon öner
                if hasattr(self, 'label_encoders') and 'recommended_vacation' in self.label_encoders:
                    fallback_class = random.randint(0, len(self.label_encoders['recommended_vacation'].classes_) - 1)
                    fallback_destination = self.label_encoders['recommended_vacation'].classes_[fallback_class]
                    
                    logger.warning(f"Tahmin hatası nedeniyle fallback mekanizması kullanıldı. Önerilen: {fallback_destination}")
                    
                    return {
                        "destination": fallback_destination,
                        "confidence": 0.1,  # Düşük güven değeri
                        "fallback": True
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback mekanizması hatası: {str(fallback_error)}")
            
            return None
    
    def _process_user_preferences(self, user_preferences):
        # Kullanıcı tercihlerini ön işle
        features = {}
        
        # Kategorik değişkenler için encoding
        for col in self.label_encoders:
            if col == 'recommended_vacation':
                # Bu tahmin edilecek değer, kullanıcı tercihlerinde olmamalı
                continue
            elif col in user_preferences:
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
                        elif 'Kültür' in known_activities:
                            features[col] = self.label_encoders[col].transform(['Kültür'])[0]
                        else:
                            # İlk sınıfı kullan
                            features[col] = 0
                    elif col == 'season':
                        # Bilinmeyen sezon için fallback
                        known_seasons = self.label_encoders[col].classes_
                        if 'Yaz' in known_seasons:
                            features[col] = self.label_encoders[col].transform(['Yaz'])[0]
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
        # Kullanıcı memnuniyeti ve değer skoru için daha yüksek varsayılan değerler
        numerical_data['value_score'] = 4.0  # Daha yüksek varsayılan değer (önceki 3.5)
        numerical_data['user_satisfaction'] = 4.5  # Daha yüksek varsayılan değer (önceki 4.0)
        
        # Sayısal verileri ölçeklendir
        numerical_features = np.array([[
            numerical_data['budget'],
            numerical_data['duration'],
            numerical_data['value_score'],
            numerical_data['user_satisfaction']
        ]], dtype=np.float64)
        
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
        instance = np.zeros(len(self.feature_columns), dtype=np.float64)
        feature_indices = {feature: i for i, feature in enumerate(self.feature_columns)}
        
        # Kategorik değerleri yerleştir
        for col, value in features.items():
            if col in feature_indices:
                instance[feature_indices[col]] = float(value)
        
        # Sayısal değerleri yerleştir
        for i, col in enumerate(['budget', 'duration', 'value_score', 'user_satisfaction']):
            if col in feature_indices:
                instance[feature_indices[col]] = float(scaled_numerical[0, i])
        
        # Bellek düzenini garanti et
        instance = np.ascontiguousarray(instance)
        
        return instance
    
    def save_model(self):
        """Modeli kaydet"""
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Özellik ağırlıklarını kaydet
            joblib.dump(self.feature_weights, 'saved_models/genetic_model.joblib')
            
            # Label encoder'ları kaydet
            joblib.dump(self.label_encoders, 'saved_models/genetic_label_encoders.joblib')
            
            # Scaler'ı kaydet
            joblib.dump(self.scaler, 'saved_models/genetic_scaler.joblib')
            
            # Özellik sütunlarını kaydet
            joblib.dump(self.feature_columns, 'saved_models/genetic_feature_columns.joblib')
            
            # Sınıf sayısını kaydet
            joblib.dump(self.num_classes, 'saved_models/genetic_num_classes.joblib')
            
            # En iyi bireyi kaydet
            if self.best_individual is not None:
                joblib.dump(self.best_individual, 'saved_models/genetic_best_individual.joblib')
            
            # Fitness geçmişini kaydet
            if hasattr(self, 'fitness_history') and self.fitness_history:
                joblib.dump(self.fitness_history, 'saved_models/genetic_fitness_history.joblib')
            
            logger.info("Genetik Algoritma modeli başarıyla kaydedildi")
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
            # Özellik ağırlıklarını yükle
            self.feature_weights = joblib.load('saved_models/genetic_model.joblib')
            
            # Label encoder'ları yükle
            self.label_encoders = joblib.load('saved_models/genetic_label_encoders.joblib')
            
            # Scaler'ı yükle
            self.scaler = joblib.load('saved_models/genetic_scaler.joblib')
            
            # Özellik sütunlarını yükle
            self.feature_columns = joblib.load('saved_models/genetic_feature_columns.joblib')
            
            # Sınıf sayısını yükle
            self.num_classes = joblib.load('saved_models/genetic_num_classes.joblib')
            
            # En iyi bireyi yükle
            try:
                self.best_individual = joblib.load('saved_models/genetic_best_individual.joblib')
            except:
                logger.warning("En iyi birey yüklenemedi.")
            
            # Fitness geçmişini yükle
            try:
                self.fitness_history = joblib.load('saved_models/genetic_fitness_history.joblib')
            except:
                logger.warning("Fitness geçmişi yüklenemedi.")
            
            logger.info("Genetik Algoritma modeli başarıyla yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
