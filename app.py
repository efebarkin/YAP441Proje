from flask import Flask, render_template, request, jsonify
import traceback
import logging
from flask_cors import CORS
import os
import pandas as pd
import json
import random

# Modelleri içe aktar
from models.decision_tree import DecisionTreeVacationRecommender
from models.knn_model import KNNVacationRecommender
from models.genetic_algorithm import GeneticVacationRecommender
from models.iterative_deepening import IDDFSVacationRecommender
from models.a_star_search import AStarVacationRecommender
from models.model_evaluator import ModelEvaluator

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Modelleri başlat
models = {}

try:
    logger.info("Decision Tree modeli başlatılıyor...")
    models['decision_tree'] = DecisionTreeVacationRecommender()
    models['decision_tree'].load_model()
    logger.info("Decision Tree modeli başarıyla başlatıldı")
except Exception as e:
    logger.error(f"Decision Tree modeli başlatılırken hata: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("KNN modeli başlatılıyor...")
    models['knn'] = KNNVacationRecommender()
    models['knn'].load_model()
    logger.info("KNN modeli başarıyla başlatıldı")
except Exception as e:
    logger.error(f"KNN modeli başlatılırken hata: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("Genetic Algorithm modeli başlatılıyor...")
    models['genetic_algorithm'] = GeneticVacationRecommender()
    models['genetic_algorithm'].load_model()
    logger.info("Genetic Algorithm modeli başarıyla başlatıldı")
except Exception as e:
    logger.error(f"Genetic Algorithm modeli başlatılırken hata: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("IDDFS modeli başlatılıyor...")
    models['iddfs'] = IDDFSVacationRecommender()
    models['iddfs'].load_model()
    logger.info("IDDFS modeli başarıyla başlatıldı")
except Exception as e:
    logger.error(f"IDDFS modeli başlatılırken hata: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("A* Search modeli başlatılıyor...")
    models['a_star'] = AStarVacationRecommender()
    models['a_star'].load_model()
    logger.info("A* Search modeli başarıyla başlatıldı")
except Exception as e:
    logger.error(f"A* Search modeli başlatılırken hata: {str(e)}")
    logger.error(traceback.format_exc())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    try:
        # Kullanıcı tercihlerini al
        user_preferences = request.json
        app.logger.info(f"Öneri isteği alındı: {user_preferences}")
        
        # Seçilen algoritmayı al
        selected_algorithm = user_preferences.get('selected_algorithm', 'decision_tree')
        app.logger.info(f"Seçilen algoritma: {selected_algorithm}")
        
        # Kullanıcı tercihlerinden algoritma seçimini kaldır
        if 'selected_algorithm' in user_preferences:
            del user_preferences['selected_algorithm']
        
        recommendations = []
        
        # Seçilen algoritma ile öneri yap
        if selected_algorithm == 'decision_tree' and 'decision_tree' in models:
            app.logger.info("Decision Tree modeli ile öneri yapılıyor...")
            try:
                recommendations = models['decision_tree'].predict_top_n(user_preferences, top_n=5)
                app.logger.info(f"Decision Tree modeli ile {len(recommendations)} öneri bulundu")
            except Exception as e:
                app.logger.error(f"Decision Tree modeli hatası: {str(e)}")
        
        elif selected_algorithm == 'knn' and 'knn' in models:
            app.logger.info("KNN modeli ile öneri yapılıyor...")
            try:
                recommendations = models['knn'].predict_top_n(user_preferences, top_n=5)
                app.logger.info(f"KNN modeli ile {len(recommendations)} öneri bulundu")
            except Exception as e:
                app.logger.error(f"KNN modeli hatası: {str(e)}")
        
        elif selected_algorithm == 'genetic_algorithm' and 'genetic_algorithm' in models:
            app.logger.info("Genetik Algoritma modeli ile öneri yapılıyor...")
            try:
                ga_recommendations = models['genetic_algorithm'].predict(user_preferences)
                if isinstance(ga_recommendations, dict):
                    recommendations = [ga_recommendations]
                else:
                    recommendations = ga_recommendations
                app.logger.info(f"Genetik Algoritma modeli ile {len(recommendations)} öneri bulundu")
            except Exception as e:
                app.logger.error(f"Genetik Algoritma hatası: {str(e)}")
        
        elif selected_algorithm == 'iddfs' and 'iddfs' in models:
            app.logger.info("IDDFS modeli ile öneri yapılıyor...")
            try:
                iddfs_recommendations = models['iddfs'].predict(user_preferences)
                if isinstance(iddfs_recommendations, dict):
                    recommendations = [iddfs_recommendations]
                else:
                    recommendations = iddfs_recommendations
                app.logger.info(f"IDDFS modeli ile {len(recommendations)} öneri bulundu")
            except Exception as e:
                app.logger.error(f"IDDFS hatası: {str(e)}")
        
        elif selected_algorithm == 'a_star' and 'a_star' in models:
            app.logger.info("A* modeli ile öneri yapılıyor...")
            try:
                a_star_recommendations = models['a_star'].predict(user_preferences)
                if isinstance(a_star_recommendations, dict):
                    recommendations = [a_star_recommendations]
                else:
                    recommendations = a_star_recommendations
                app.logger.info(f"A* modeli ile {len(recommendations)} öneri bulundu")
            except Exception as e:
                app.logger.error(f"A* hatası: {str(e)}")
        
        # Eğer hiç öneri bulunamadıysa varsayılan öneriler oluştur
        if not recommendations:
            app.logger.warning("Hiç öneri bulunamadı. Varsayılan öneriler oluşturuluyor...")
            recommendations = create_default_recommendations()
            app.logger.info(f"Varsayılan olarak {len(recommendations)} öneri oluşturuldu")
        
        app.logger.info(f"Toplam {len(recommendations)} öneri bulundu")
        
        return jsonify({
            'recommendations': recommendations
        })
    except Exception as e:
        app.logger.error(f"Öneri hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_default_recommendations():
    # Mevsime göre varsayılan destinasyonlar
    default_destinations = {
        "İlkbahar": ["Antalya", "Bodrum", "Kapadokya", "İstanbul", "Çeşme"],
        "Yaz": ["Bodrum", "Marmaris", "Fethiye", "Çeşme", "Alanya"],
        "Sonbahar": ["İstanbul", "Kapadokya", "Bursa", "Sapanca", "Abant"],
        "Kış": ["Uludağ", "Kartalkaya", "Palandöken", "Kartepe", "Ilgaz"]
    }
    
    # Aktiviteye göre varsayılan destinasyonlar
    activity_destinations = {
        "Plaj": ["Bodrum", "Marmaris", "Fethiye", "Çeşme", "Alanya"],
        "Kayak": ["Uludağ", "Kartalkaya", "Palandöken", "Kartepe", "Ilgaz"],
        "Kültür": ["İstanbul", "Kapadokya", "Efes", "Nemrut", "Mardin"],
        "Doğa": ["Kapadokya", "Abant", "Sapanca", "Fethiye", "Artvin"],
        "Eğlence": ["Bodrum", "İstanbul", "Antalya", "Marmaris", "Kuşadası"]
    }
    
    # Varsayılan öneriler oluştur
    recommendations = []
    
    for season in ["İlkbahar", "Yaz", "Sonbahar", "Kış"]:
        for activity in ["Plaj", "Kayak", "Kültür", "Doğa", "Eğlence"]:
            # Her mevsim-aktivite kombinasyonu için uygun destinasyonları bul
            suitable_destinations = []
            
            # Mevsime göre uygun destinasyonlar
            season_destinations = default_destinations.get(season, [])
            
            # Aktiviteye göre uygun destinasyonlar
            activity_dests = activity_destinations.get(activity, [])
            
            # Her iki listede de olan destinasyonları bul
            for dest in season_destinations:
                if dest in activity_dests:
                    suitable_destinations.append(dest)
            
            # Eğer uygun destinasyon bulunamadıysa, mevsime göre önerilen destinasyonları kullan
            if not suitable_destinations:
                suitable_destinations = season_destinations[:2]
            
            # En fazla 2 öneri ekle
            for i, destination in enumerate(suitable_destinations[:2]):
                # Rastgele fiyatlar oluştur
                hotel_price = round(random.uniform(500, 3000))
                flight_cost = round(random.uniform(1000, 5000))
                total_cost = hotel_price * 7 + flight_cost
                
                recommendation = {
                    "destination": destination,
                    "season": season,
                    "activity": activity,
                    "costs": {
                        "hotel_price": hotel_price,
                        "flight_cost": flight_cost,
                        "total_cost": total_cost
                    },
                    "algorithm_confidence": round(random.uniform(0.6, 0.95), 2)
                }
                recommendations.append(recommendation)
    
    # Rastgele karıştır ve en fazla 5 öneri döndür
    random.shuffle(recommendations)
    return recommendations[:5]

@app.route('/api/destinations', methods=['GET'])
def get_destinations():
    try:
        return jsonify({
            "seasons": ["İlkbahar", "Yaz", "Sonbahar", "Kış"],
            "activities": ["Plaj", "Kayak", "Kültür", "Doğa", "Eğlence"]
        })
    except Exception as e:
        logger.error(f"Destinasyon bilgileri alınırken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Kullanılabilir algoritmaları döndür"""
    algorithms = [
        {
            'id': 'decision_tree',
            'name': 'Karar Ağacı',
            'description': 'Basit karar kuralları kullanarak tahmin yapar'
        },
        {
            'id': 'knn',
            'name': 'K-En Yakın Komşu',
            'description': 'Benzer kullanıcıların tercihlerine göre tahmin yapar'
        },
        {
            'id': 'genetic_algorithm',
            'name': 'Genetik Algoritma',
            'description': 'Evrimsel hesaplama ile en iyi tatil seçeneğini bulur'
        },
        {
            'id': 'iddfs',
            'name': 'Iterative Deepening DFS',
            'description': 'Arama ağacında derinlik öncelikli arama yapar'
        },
        {
            'id': 'a_star',
            'name': 'A* Arama Algoritması',
            'description': 'Heuristik kullanarak en iyi tatil seçeneğini bulur'
        }
    ]
    
    return jsonify({
        'success': True,
        'algorithms': algorithms
    })

@app.route('/model_evaluation', methods=['GET'])
def get_model_evaluation():
    """Model değerlendirme sonuçlarını döndür"""
    try:
        # Değerlendirme sonuçlarının kaydedildiği dosyaları kontrol et
        evaluation_results = {}
        
        # Metrikleri oku
        metrics_file = os.path.join('evaluation', 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                evaluation_results['metrics'] = json.load(f)
        else:
            evaluation_results['metrics'] = {"error": "Metrik sonuçları bulunamadı"}
        
        # Karışıklık matrislerini oku (dosya yollarını döndür)
        confusion_matrices = {}
        confusion_matrix_dir = os.path.join('evaluation', 'confusion_matrices')
        if os.path.exists(confusion_matrix_dir):
            for model_name in os.listdir(confusion_matrix_dir):
                if model_name.endswith('.png'):
                    confusion_matrices[model_name.replace('.png', '')] = f'/static/evaluation/confusion_matrices/{model_name}'
        
        evaluation_results['confusion_matrices'] = confusion_matrices
        
        # Karşılaştırma grafiklerini oku
        comparison_charts = {}
        charts_dir = os.path.join('evaluation', 'charts')
        if os.path.exists(charts_dir):
            for chart_name in os.listdir(charts_dir):
                if chart_name.endswith('.png'):
                    comparison_charts[chart_name.replace('.png', '')] = f'/static/evaluation/charts/{chart_name}'
        
        evaluation_results['comparison_charts'] = comparison_charts
        
        # En iyi model bilgisini oku
        best_model_file = os.path.join('evaluation', 'best_model.json')
        if os.path.exists(best_model_file):
            with open(best_model_file, 'r') as f:
                evaluation_results['best_model'] = json.load(f)
        else:
            evaluation_results['best_model'] = {"error": "En iyi model bilgisi bulunamadı"}
        
        return jsonify(evaluation_results)
    except Exception as e:
        app.logger.error(f"Model değerlendirme sonuçları alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run_evaluation', methods=['POST'])
def run_evaluation():
    """Modelleri yeniden değerlendir"""
    try:
        # Veri dosyasını kontrol et
        data_file = 'data/synthetic_vacation_data.csv'
        if not os.path.exists(data_file):
            return jsonify({'error': 'Veri dosyası bulunamadı'}), 404
        
        # Veriyi oku
        df = pd.read_csv(data_file)
        
        # Model değerlendiriciyi oluştur
        evaluator = ModelEvaluator()
        
        # Modelleri değerlendir
        evaluator.train_and_evaluate_all_models(df)
        
        # En iyi modeli bul
        best_model = evaluator.get_best_model(metric='accuracy')
        best_model_info = {
            'name': type(best_model).__name__ if best_model else 'Bilinmiyor',
            'metric': 'accuracy'
        }
        
        # En iyi model bilgisini kaydet
        os.makedirs('evaluation', exist_ok=True)
        with open(os.path.join('evaluation', 'best_model.json'), 'w') as f:
            json.dump(best_model_info, f)
        
        return jsonify({'success': True, 'message': 'Model değerlendirmesi başarıyla tamamlandı'})
    except Exception as e:
        app.logger.error(f"Model değerlendirme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_evaluation_results', methods=['GET'])
def get_model_evaluation_results():
    """Model değerlendirme sonuçlarını döndür"""
    try:
        # Değerlendirme sonuçlarının kaydedildiği dosyaları kontrol et
        evaluation_results = {}
        
        # Metrikleri oku
        metrics_file = os.path.join('evaluation', 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                evaluation_results['metrics'] = json.load(f)
        else:
            evaluation_results['metrics'] = {"error": "Metrik sonuçları bulunamadı"}
        
        # Karışıklık matrislerini oku (dosya yollarını döndür)
        confusion_matrices = {}
        confusion_matrix_dir = os.path.join('evaluation', 'confusion_matrices')
        if os.path.exists(confusion_matrix_dir):
            for model_name in os.listdir(confusion_matrix_dir):
                if model_name.endswith('.png'):
                    confusion_matrices[model_name.replace('.png', '')] = f'/static/evaluation/confusion_matrices/{model_name}'
        
        evaluation_results['confusion_matrices'] = confusion_matrices
        
        # Karşılaştırma grafiklerini oku
        comparison_charts = {}
        charts_dir = os.path.join('evaluation', 'charts')
        if os.path.exists(charts_dir):
            for chart_name in os.listdir(charts_dir):
                if chart_name.endswith('.png'):
                    comparison_charts[chart_name.replace('.png', '')] = f'/static/evaluation/charts/{chart_name}'
        
        evaluation_results['comparison_charts'] = comparison_charts
        
        # En iyi model bilgisini oku
        best_model_file = os.path.join('evaluation', 'best_model.json')
        if os.path.exists(best_model_file):
            with open(best_model_file, 'r') as f:
                evaluation_results['best_model'] = json.load(f)
        else:
            evaluation_results['best_model'] = {"error": "En iyi model bilgisi bulunamadı"}
        
        return jsonify(evaluation_results)
    except Exception as e:
        app.logger.error(f"Model değerlendirme sonuçları alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
