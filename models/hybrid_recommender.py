import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import LabelEncoder

# Modelleri içe aktar
from models.decision_tree import DecisionTreeVacationRecommender
from models.knn_model import KNNVacationRecommender
from models.genetic_algorithm import GeneticVacationRecommender
from models.iterative_deepening import IDDFSVacationRecommender
from models.a_star_search import AStarVacationRecommender

logger = logging.getLogger(__name__)

class HybridVacationRecommender:
    def __init__(self):
        """Hibrit öneri sistemi için modelleri yükle"""
        self.models = {
            'decision_tree': None,
            'knn': None,
            'genetic_algorithm': None,
            'iddfs': None,
            'a_star': None
        }
        
        # Modelleri yükle
        try:
            self.models['decision_tree'] = DecisionTreeVacationRecommender()
            self.models['decision_tree'].load_model()
            logger.info("Decision Tree modeli yüklendi")
        except Exception as e:
            logger.error(f"Decision Tree modeli yüklenirken hata: {str(e)}")
        
        try:
            self.models['knn'] = KNNVacationRecommender()
            self.models['knn'].load_model()
            logger.info("KNN modeli yüklendi")
        except Exception as e:
            logger.error(f"KNN modeli yüklenirken hata: {str(e)}")
        
        try:
            self.models['genetic_algorithm'] = GeneticVacationRecommender()
            self.models['genetic_algorithm'].load_model()
            logger.info("Genetic Algorithm modeli yüklendi")
        except Exception as e:
            logger.error(f"Genetic Algorithm modeli yüklenirken hata: {str(e)}")
        
        try:
            self.models['iddfs'] = IDDFSVacationRecommender()
            self.models['iddfs'].load_model()
            logger.info("IDDFS modeli yüklendi")
        except Exception as e:
            logger.error(f"IDDFS modeli yüklenirken hata: {str(e)}")
        
        try:
            self.models['a_star'] = AStarVacationRecommender()
            self.models['a_star'].load_model()
            logger.info("A* Search modeli yüklendi")
        except Exception as e:
            logger.error(f"A* Search modeli yüklenirken hata: {str(e)}")
        
        # Label encoder'ları yükle
        try:
            self.encoders = joblib.load('models/label_encoders.pkl')
            logger.info("Label encoder'lar yüklendi")
        except Exception as e:
            logger.error(f"Label encoder'lar yüklenirken hata: {str(e)}")
            self.encoders = {}
    
    def recommend(self, user_preferences, algorithm='decision_tree', top_n=5):
        """Kullanıcı tercihlerine göre tatil önerileri yap
        
        Args:
            user_preferences (dict): Kullanıcı tercihleri (budget, duration, season, preferred_activity)
            algorithm (str): Kullanılacak algoritma ('decision_tree', 'knn', 'genetic_algorithm', 'iddfs', 'a_star')
            top_n (int): Döndürülecek öneri sayısı
            
        Returns:
            list: Öneri listesi
        """
        logger.info(f"Öneri isteniyor. Algoritma: {algorithm}, Tercihler: {user_preferences}")
        
        if algorithm not in self.models or self.models[algorithm] is None:
            logger.error(f"Belirtilen algoritma ({algorithm}) bulunamadı veya yüklenemedi.")
            # Varsayılan olarak mevcut ilk algoritmayı kullan
            for alg_name, model in self.models.items():
                if model is not None:
                    algorithm = alg_name
                    logger.info(f"Varsayılan algoritma olarak {algorithm} kullanılacak")
                    break
            else:
                logger.error("Hiçbir model yüklenemedi.")
                return []
        
        try:
            # Seçilen algoritmaya göre öneri yap
            model = self.models[algorithm]
            
            # Algoritma tipine göre farklı işlemler
            if algorithm in ['decision_tree', 'knn']:
                # Supervised learning modelleri için
                recommendations = model.predict_top_n(user_preferences, top_n)
            elif algorithm in ['genetic_algorithm', 'iddfs', 'a_star']:
                # Klasik AI algoritmaları için
                recommendations = model.predict(user_preferences, top_n)
            else:
                logger.error(f"Bilinmeyen algoritma tipi: {algorithm}")
                return []
            
            # Sonuçları formatla
            formatted_recommendations = []
            for rec in recommendations:
                if rec is not None:
                    # Temel bilgileri ekle
                    formatted_rec = {
                        'destination': rec['destination'],
                        'confidence': rec.get('confidence', 0.8),  # Varsayılan güven değeri
                        'season': rec.get('season', user_preferences.get('season')),
                        'activity': rec.get('activity', user_preferences.get('preferred_activity')),
                    }
                    
                    # Maliyet bilgilerini ekle (varsa)
                    if 'costs' in rec:
                        formatted_rec['costs'] = rec['costs']
                    else:
                        # Varsayılan maliyet bilgileri
                        budget = user_preferences.get('budget', 10000)
                        duration = user_preferences.get('duration', 7)
                        
                        # Basit bir hesaplama
                        hotel_price = budget * 0.6 / duration
                        flight_cost = budget * 0.3
                        total_cost = hotel_price * duration + flight_cost
                        
                        formatted_rec['costs'] = {
                            'hotel_price': hotel_price,
                            'flight_cost': flight_cost,
                            'total_cost': total_cost
                        }
                    
                    # Algoritma spesifik güven değerlerini ekle
                    formatted_rec['algorithm_confidence'] = formatted_rec['confidence']
                    
                    formatted_recommendations.append(formatted_rec)
            
            logger.info(f"{len(formatted_recommendations)} öneri bulundu.")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Öneri yapılırken hata: {str(e)}")
            return []
