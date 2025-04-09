import pandas as pd
import numpy as np
import logging
import os
from models.decision_tree import DecisionTreeVacationRecommender
from models.knn_model import KNNVacationRecommender
from models.iterative_deepening import IDDFSVacationRecommender
from models.genetic_algorithm import GeneticVacationRecommender
from models.a_star_search import AStarVacationRecommender
from models.model_evaluator import ModelEvaluator
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışması için
import time

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data():
    """Sentetik veri üret"""
    logger.info("Veri üretimi başlıyor...")
    from models.veri_uretimi import generate_vacation_data
    
    # Veri dizini kontrolü
    if not os.path.exists('data'):
        os.makedirs('data')
        logger.info("'data' dizini oluşturuldu")
    
    # Daha fazla veri üret - 5000 yerine 10000 örnek
    df = generate_vacation_data(n_samples=10000)
    
    # Veriyi kaydet
    df.to_csv('data/synthetic_vacation_data.csv', index=False)
    logger.info("Veri üretildi ve kaydedildi: data/synthetic_vacation_data.csv")
    
    return df

def train_decision_tree(df):
    """Karar Ağacı modelini eğit"""
    logger.info("Karar Ağacı model eğitimi başlıyor...")
    
    # Karar Ağacı model nesnesini oluştur
    dt_model = DecisionTreeVacationRecommender()
    
    # Veri dengesizliğini kontrol et
    destination_counts = df['destination'].value_counts()
    logger.info(f"Destinasyon dağılımı: {destination_counts}")
    
    # Veriyi dengele - az örnekli sınıflardan daha fazla örnek al
    min_samples = destination_counts.min()
    balanced_dfs = []
    
    for dest in destination_counts.index:
        dest_df = df[df['destination'] == dest]
        # Az örnekli sınıflar için daha fazla örnek al
        if len(dest_df) < min_samples * 2:
            # Örnekleri çoğalt
            multiplier = int(min_samples * 2 / len(dest_df)) + 1
            dest_df = pd.concat([dest_df] * multiplier)
        
        # Rastgele örnekle
        if len(dest_df) > min_samples * 2:
            dest_df = dest_df.sample(min_samples * 2, replace=False)
        
        balanced_dfs.append(dest_df)
    
    balanced_df = pd.concat(balanced_dfs)
    logger.info(f"Dengeli veri seti boyutu: {len(balanced_df)}")
    
    # Modeli eğit
    dt_model.train(balanced_df)
    
    # Modeli kaydet
    dt_model.save_model()
    
    logger.info("Karar Ağacı model eğitimi tamamlandı")

def train_knn(df):
    """KNN modelini eğit"""
    logger.info("KNN model eğitimi başlıyor...")
    
    # KNN model nesnesini oluştur
    knn_model = KNNVacationRecommender()
    
    # Veri dengesizliğini kontrol et
    destination_counts = df['destination'].value_counts()
    logger.info(f"Destinasyon dağılımı: {destination_counts}")
    
    # Veriyi dengele - az örnekli sınıflardan daha fazla örnek al
    min_samples = destination_counts.min()
    balanced_dfs = []
    
    for dest in destination_counts.index:
        dest_df = df[df['destination'] == dest]
        # Az örnekli sınıflar için daha fazla örnek al
        if len(dest_df) < min_samples * 2:
            # Örnekleri çoğalt
            multiplier = int(min_samples * 2 / len(dest_df)) + 1
            dest_df = pd.concat([dest_df] * multiplier)
        
        # Rastgele örnekle
        if len(dest_df) > min_samples * 2:
            dest_df = dest_df.sample(min_samples * 2, replace=False)
        
        balanced_dfs.append(dest_df)
    
    balanced_df = pd.concat(balanced_dfs)
    logger.info(f"Dengeli veri seti boyutu: {len(balanced_df)}")
    
    # Modeli eğit
    knn_model.train(balanced_df)
    
    # Modeli kaydet
    knn_model.save_model()
    
    logger.info("KNN model eğitimi tamamlandı")

def train_iddfs(df):
    """IDDFS modelini eğit"""
    logger.info("IDDFS model eğitimi başlıyor...")
    
    # IDDFS model nesnesini oluştur
    iddfs_model = IDDFSVacationRecommender()
    
    # Veri dengesizliğini kontrol et
    destination_counts = df['destination'].value_counts()
    logger.info(f"Destinasyon dağılımı: {destination_counts}")
    
    # Veriyi dengele - az örnekli sınıflardan daha fazla örnek al
    min_samples = destination_counts.min()
    balanced_dfs = []
    
    for dest in destination_counts.index:
        dest_df = df[df['destination'] == dest]
        # Az örnekli sınıflar için daha fazla örnek al
        if len(dest_df) < min_samples * 2:
            # Örnekleri çoğalt
            multiplier = int(min_samples * 2 / len(dest_df)) + 1
            dest_df = pd.concat([dest_df] * multiplier)
        
        # Rastgele örnekle
        if len(dest_df) > min_samples * 2:
            dest_df = dest_df.sample(min_samples * 2, replace=False)
        
        balanced_dfs.append(dest_df)
    
    balanced_df = pd.concat(balanced_dfs)
    logger.info(f"Dengeli veri seti boyutu: {len(balanced_df)}")
    
    # Modeli geliştirilmiş parametrelerle eğit
    # Daha düşük benzerlik eşiği ve daha yüksek derinlik kullan
    iddfs_model.train(balanced_df, max_depth=20, similarity_threshold=0.2)
    
    # Modeli kaydet
    iddfs_model.save_model()
    
    logger.info("IDDFS model eğitimi tamamlandı")

def train_genetic_algorithm(df):
    """Genetik Algoritma modelini eğit"""
    logger.info("Genetik Algoritma model eğitimi başlıyor...")
    
    # Genetik Algoritma model nesnesini oluştur
    ga_model = GeneticVacationRecommender()
    
    # Veri dengesizliğini kontrol et
    destination_counts = df['destination'].value_counts()
    logger.info(f"Destinasyon dağılımı: {destination_counts}")
    
    # Veriyi dengele - az örnekli sınıflardan daha fazla örnek al
    min_samples = destination_counts.min()
    balanced_dfs = []
    
    for dest in destination_counts.index:
        dest_df = df[df['destination'] == dest]
        # Az örnekli sınıflar için daha fazla örnek al
        if len(dest_df) < min_samples * 2:
            # Örnekleri çoğalt
            multiplier = int(min_samples * 2 / len(dest_df)) + 1
            dest_df = pd.concat([dest_df] * multiplier)
        
        # Rastgele örnekle
        if len(dest_df) > min_samples * 2:
            dest_df = dest_df.sample(min_samples * 2, replace=False)
        
        balanced_dfs.append(dest_df)
    
    balanced_df = pd.concat(balanced_dfs)
    logger.info(f"Dengeli veri seti boyutu: {len(balanced_df)}")
    
    # Modeli geliştirilmiş parametrelerle eğit
    ga_model.train(balanced_df, population_size=150, generations=100, mutation_rate=0.15, tournament_size=7)
    
    # Modeli kaydet
    ga_model.save_model()
    
    logger.info("Genetik Algoritma model eğitimi tamamlandı")

def evaluate_all_models_without_training(df):
    """Tüm modelleri eğitmeden değerlendir"""
    logger.info("Eğitilmiş modellerin değerlendirmesi başlıyor...")
    
    # Modelleri yükle
    models = {}
    
    # Decision Tree
    try:
        dt_model = DecisionTreeVacationRecommender()
        dt_model.load_model()
        models['Decision Tree'] = dt_model
        logger.info("Decision Tree modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"Decision Tree modeli yüklenirken hata: {str(e)}")
    
    # KNN
    try:
        knn_model = KNNVacationRecommender()
        knn_model.load_model()
        models['KNN'] = knn_model
        logger.info("KNN modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"KNN modeli yüklenirken hata: {str(e)}")
    
    # IDDFS
    try:
        iddfs_model = IDDFSVacationRecommender()
        iddfs_model.load_model()
        models['IDDFS'] = iddfs_model
        logger.info("IDDFS modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"IDDFS modeli yüklenirken hata: {str(e)}")
    
    # Genetic Algorithm
    try:
        ga_model = GeneticVacationRecommender()
        ga_model.load_model()
        models['Genetic Algorithm'] = ga_model
        logger.info("Genetic Algorithm modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"Genetic Algorithm modeli yüklenirken hata: {str(e)}")
    
    # A* Search
    try:
        astar_model = AStarVacationRecommender()
        astar_model.load_model()
        models['A* Search'] = astar_model
        logger.info("A* Search modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"A* Search modeli yüklenirken hata: {str(e)}")
    
    # Değerlendirme için küçük bir test seti kullan
    # Uyarıları ve KeyboardInterrupt hatalarını önlemek için test setini küçült
    test_size = min(100, len(df) // 10)  # Veri setinin %10'u veya en fazla 100 örnek
    test_df = df.sample(n=test_size, random_state=42)
    
    # Değerlendirme için özel bir fonksiyon kullan
    evaluate_models_manually(models, test_df)
    
    logger.info("Tüm modellerin değerlendirmesi tamamlandı!")

def evaluate_models_manually(models, test_df):
    """Modelleri manuel olarak değerlendir (scikit-learn uyarılarını önlemek için)"""
    logger.info("Model değerlendirmesi başlıyor...")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"{model_name} değerlendirmesi başlıyor...")
        
        try:
            # Değerlendirme için zamanı ölç
            start_time = time.time()
            
            # Tahminler
            predictions = []
            ground_truth = []
            confidences = []
            
            # Her örnek için tahmin yap
            for _, row in test_df.iterrows():
                user_preferences = {
                    'season': row['season'],
                    'preferred_activity': row['preferred_activity'],
                    'budget': row['budget'],
                    'duration': row['duration']
                }
                
                try:
                    # Tahmin yap
                    predictions_result = model.predict(user_preferences, top_n=5)
                    
                    # Sonuçları kontrol et
                    if predictions_result and len(predictions_result) > 0:
                        # İlk öneriyi al
                        top_prediction = predictions_result[0]
                        
                        # Sonuç formatını kontrol et
                        if isinstance(top_prediction, dict) and 'destination' in top_prediction:
                            predictions.append(top_prediction['destination'])
                            confidences.append(top_prediction.get('confidence', 0.0))
                            ground_truth.append(row['recommended_vacation'])
                        elif isinstance(predictions_result, dict) and 'destination' in predictions_result:
                            predictions.append(predictions_result['destination'])
                            confidences.append(predictions_result.get('confidence', 0.0))
                            ground_truth.append(row['recommended_vacation'])
                except Exception as e:
                    logger.warning(f"{model_name} için tahmin hatası: {str(e)}")
                    continue
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Metrikleri hesapla
            if len(predictions) > 0:
                # Doğruluk
                correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
                accuracy = correct / len(predictions)
                
                # Precision, Recall ve F1 (manuel hesapla)
                labels = list(set(ground_truth + predictions))
                precision_sum = 0
                recall_sum = 0
                f1_sum = 0
                label_count = 0
                
                for label in labels:
                    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p == g and p == label)
                    false_positives = sum(1 for p, g in zip(predictions, ground_truth) if p != g and p == label)
                    false_negatives = sum(1 for p, g in zip(predictions, ground_truth) if p != g and g == label)
                    
                    # Sıfıra bölme hatasını önle
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precision_sum += precision
                    recall_sum += recall
                    f1_sum += f1
                    label_count += 1
                
                # Ağırlıklı ortalama
                precision = precision_sum / label_count if label_count > 0 else 0
                recall = recall_sum / label_count if label_count > 0 else 0
                f1 = f1_sum / label_count if label_count > 0 else 0
                
                # Ortalama güven
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Sonuçları kaydet
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'avg_confidence': avg_confidence,
                    'inference_time': inference_time / len(predictions) if len(predictions) > 0 else 0,
                    'predictions': len(predictions)
                }
                
                logger.info(f"{model_name} değerlendirmesi tamamlandı.")
                logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                logger.info(f"Ortalama güven: {avg_confidence:.4f}, Çıkarım süresi: {inference_time/len(predictions):.6f} saniye/örnek")
            else:
                logger.warning(f"{model_name} için tahmin yapılamadı.")
                
        except Exception as e:
            logger.error(f"{model_name} değerlendirmesi sırasında hata: {str(e)}")
    
    # En iyi modeli bul
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"En iyi model (doğruluk metriğine göre): {best_model[0]}")
    
    return results

def train_a_star(df):
    """A* Search modelini eğit"""
    logger.info("A* Search model eğitimi başlıyor...")
    
    # A* Search model nesnesini oluştur
    a_star_model = AStarVacationRecommender()
    
    # Veri dengesizliğini kontrol et
    destination_counts = df['destination'].value_counts()
    logger.info(f"Destinasyon dağılımı: {destination_counts}")
    
    # Veriyi dengele - az örnekli sınıflardan daha fazla örnek al
    min_samples = destination_counts.min()
    balanced_dfs = []
    
    for dest in destination_counts.index:
        dest_df = df[df['destination'] == dest]
        # Az örnekli sınıflar için daha fazla örnek al
        if len(dest_df) < min_samples * 2:
            # Örnekleri çoğalt
            multiplier = int(min_samples * 2 / len(dest_df)) + 1
            dest_df = pd.concat([dest_df] * multiplier)
        
        # Rastgele örnekle
        if len(dest_df) > min_samples * 2:
            dest_df = dest_df.sample(min_samples * 2, replace=False)
        
        balanced_dfs.append(dest_df)
    
    balanced_df = pd.concat(balanced_dfs)
    logger.info(f"Dengeli veri seti boyutu: {len(balanced_df)}")
    
    # Modeli eğit (iterations ve learning_rate parametrelerini kaldırdık)
    a_star_model.train(balanced_df)
    
    # Modeli kaydet
    a_star_model.save_model()
    
    logger.info("A* Search model eğitimi tamamlandı")

def train_all_models_with_evaluator(df):
    """Tüm modelleri eğit ve karşılaştır"""
    logger.info("Tüm modellerin eğitimi ve karşılaştırması başlıyor...")
    
    # Model değerlendirici oluştur
    evaluator = ModelEvaluator()
    
    # Modelleri eğit ve değerlendir
    evaluator.train_and_evaluate_all_models(df)
    
    # En iyi modeli bul ve logla
    best_model = evaluator.get_best_model(metric='accuracy')
    if best_model:
        logger.info(f"En iyi model (doğruluk metriğine göre): {type(best_model).__name__}")
    
    logger.info("Tüm modellerin eğitimi ve karşılaştırması tamamlandı")

def main():
    """Ana eğitim fonksiyonu"""
    logger.info("Model eğitimi başlıyor...")
    
    # Veri dizini kontrolü
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Model dizini kontrolü
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Sentetik veri var mı kontrol et
    if os.path.exists('data/synthetic_vacation_data.csv'):
        logger.info("Mevcut veri kullanılıyor: data/synthetic_vacation_data.csv")
        df = pd.read_csv('data/synthetic_vacation_data.csv')
        
        # Veri boyutu kontrolü - eğer 5000'den az ise yeni veri üret
        if len(df) < 1000:
            logger.info("Mevcut veri yetersiz, yeni veri üretiliyor...")
            df = generate_data()
    else:
        logger.info("Veri bulunamadı, yeni veri üretiliyor...")
        df = generate_data()

    # Tüm modelleri tek seferde eğit ve karşılaştır
    train_a_star(df)
    # train_genetic_algorithm(df)
    # train_iddfs(df)
    # train_knn(df)
    # train_decision_tree(df)
    # train_all_models_with_evaluator(df)
    
    logger.info("Tüm modellerin eğitimi tamamlandı!")

if __name__ == "__main__":
    main()
