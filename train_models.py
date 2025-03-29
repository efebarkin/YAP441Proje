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
    """Eğitilmiş modelleri yükle ve değerlendir (eğitim yapmadan)"""
    logger.info("Eğitilmiş modellerin değerlendirmesi başlıyor...")
    
    # Model değerlendirici oluştur
    evaluator = ModelEvaluator()
    
    # Modelleri yükle
    for model_name, model in evaluator.models.items():
        try:
            # Modeli yükle
            model.load_model()
            logger.info(f"{model_name} modeli başarıyla yüklendi.")
            
            # Sonuçları sakla
            evaluator.results[model_name] = {
                'model': model,
                'training_time': 0  # Eğitim yapılmadığı için 0
            }
        except Exception as e:
            logger.error(f"{model_name} modeli yüklenirken hata: {str(e)}")
    
    # Modelleri değerlendir
    evaluator.evaluate_all_models(df)
    
    # Karşılaştırma grafiği oluştur
    comparison_df = evaluator.compare_models()
    
    # En iyi modeli bul ve logla
    best_model = evaluator.get_best_model(metric='accuracy')
    if best_model:
        logger.info(f"En iyi model (doğruluk metriğine göre): {type(best_model).__name__}")
    
    logger.info("Tüm modellerin değerlendirmesi tamamlandı")
    return comparison_df

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
    
    # Modeli geliştirilmiş parametrelerle eğit
    a_star_model.train(balanced_df, iterations=200, learning_rate=0.01)
    
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

    # Modelleri değerlendir (eğitim yapmadan)
    # evaluate_all_models_without_training(df)
    
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
