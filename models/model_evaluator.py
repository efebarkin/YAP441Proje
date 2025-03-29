import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import os
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

# Modelleri içe aktar
from models.decision_tree import DecisionTreeVacationRecommender
from models.knn_model import KNNVacationRecommender
from models.iterative_deepening import IDDFSVacationRecommender
from models.genetic_algorithm import GeneticVacationRecommender
from models.a_star_search import AStarVacationRecommender

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.models = {
            'Decision Tree': DecisionTreeVacationRecommender(),
            'KNN': KNNVacationRecommender(),
            'IDDFS': IDDFSVacationRecommender(),
            'Genetic Algorithm': GeneticVacationRecommender(),
            'A* Search': AStarVacationRecommender()
        }
        self.results = {}
        
    def train_and_evaluate_all_models(self, df):
        """Tüm modelleri eğit ve değerlendir"""
        logger.info("Tüm modellerin eğitimi ve değerlendirmesi başlıyor...")
        
        # Değerlendirme dizinlerini oluştur
        os.makedirs('evaluation', exist_ok=True)
        os.makedirs('evaluation/confusion_matrices', exist_ok=True)
        os.makedirs('evaluation/charts', exist_ok=True)
        
        for model_name, model in self.models.items():
            logger.info(f"{model_name} eğitimi başlıyor...")
            start_time = time.time()
            
            try:
                model.train(df)
                end_time = time.time()
                training_time = end_time - start_time
                logger.info(f"{model_name} eğitimi tamamlandı. Süre: {training_time:.2f} saniye")
                
                # Modeli kaydet
                model.save_model()
                logger.info(f"{model_name} modeli kaydedildi.")
                
                # Sonuçları sakla
                self.results[model_name] = {
                    'training_time': training_time,
                    'model': model
                }
                
            except Exception as e:
                logger.error(f"{model_name} eğitimi sırasında hata: {str(e)}")
        
        # Modelleri değerlendir
        self.evaluate_all_models(df)
        
        # Karşılaştırma grafiği oluştur
        comparison_df = self.compare_models()
        
        # Değerlendirme sonuçlarını JSON olarak kaydet
        if comparison_df is not None and not comparison_df.empty:
            metrics_json = comparison_df.to_dict(orient='records')
            with open('evaluation/metrics.json', 'w') as f:
                json.dump(metrics_json, f)
            logger.info("Değerlendirme metrikleri JSON olarak kaydedildi: evaluation/metrics.json")
        
        logger.info("Tüm modellerin eğitimi ve değerlendirmesi tamamlandı.")
        
    def evaluate_all_models(self, df):
        """Tüm modelleri değerlendir"""
        logger.info("Tüm modellerin değerlendirmesi başlıyor...")
        
        # Test verisi oluştur - Daha büyük test seti kullan
        test_size = min(1000, len(df) // 4)  # Veri setinin %25'i veya en fazla 1000 örnek
        test_df = df.sample(n=test_size, random_state=42)
        
        for model_name, result in self.results.items():
            model = result['model']
            logger.info(f"{model_name} değerlendirmesi başlıyor...")
            
            try:
                # Değerlendirme için zamanı ölç
                start_time = time.time()
                
                # Tahminler
                predictions = []
                ground_truth = []
                confidences = []
                
                for _, row in test_df.iterrows():
                    user_preferences = {
                        'season': row['season'],
                        'preferred_activity': row['preferred_activity'],
                        'budget': row['budget'],
                        'duration': row['duration']
                    }
                    
                    try:
                        # Modelin birden fazla öneri döndürdüğünü varsayalım
                        predictions_result = model.predict(user_preferences, top_n=5)
                        
                        # Eğer sonuç None değilse ve boş değilse
                        if predictions_result and len(predictions_result) > 0:
                            # İlk öneriyi alalım (en yüksek güven skoruna sahip olan)
                            top_prediction = predictions_result[0]
                            
                            # Sonuç bir sözlük mü yoksa liste mi kontrol edelim
                            if isinstance(top_prediction, dict) and 'destination' in top_prediction:
                                predictions.append(top_prediction['destination'])
                                confidences.append(top_prediction.get('confidence', 0.0))
                                ground_truth.append(row['recommended_vacation'])
                            elif isinstance(predictions_result, dict) and 'destination' in predictions_result:
                                # Bazı modeller tek bir sözlük döndürebilir
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
                    accuracy = accuracy_score(ground_truth, predictions)
                    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
                    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
                    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    
                    # Confusion matrix oluştur
                    cm = confusion_matrix(ground_truth, predictions)
                    
                    # Sonuçları sakla
                    self.results[model_name].update({
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'avg_confidence': avg_confidence,
                        'inference_time': inference_time / len(predictions) if len(predictions) > 0 else 0,
                        'predictions': len(predictions),
                        'confusion_matrix': cm
                    })
                    
                    # Confusion matrix görselleştir
                    self._plot_confusion_matrix(cm, model_name)
                    
                    logger.info(f"{model_name} değerlendirmesi tamamlandı.")
                    logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    logger.info(f"Ortalama güven: {avg_confidence:.4f}, Çıkarım süresi: {inference_time/len(predictions):.6f} saniye/örnek")
                else:
                    logger.warning(f"{model_name} için tahmin yapılamadı.")
                    
            except Exception as e:
                logger.error(f"{model_name} değerlendirmesi sırasında hata: {str(e)}")
        
        logger.info("Tüm modellerin değerlendirmesi tamamlandı.")
        return self.results
    
    def _plot_confusion_matrix(self, cm, model_name):
        """Confusion matrix görselleştir"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            plt.ylabel('Gerçek Değer')
            plt.xlabel('Tahmin Edilen Değer')
            plt.tight_layout()
            
            # Dosya adını güvenli hale getir
            safe_model_name = model_name.lower().replace(" ", "_").replace("*", "star")
            plt.savefig(f'models/{safe_model_name}_confusion_matrix.png', dpi=300)
            logger.info(f"{model_name} confusion matrix kaydedildi")
        except Exception as e:
            logger.error(f"Confusion matrix görselleştirme hatası: {str(e)}")
    
    def compare_models(self):
        """Modelleri karşılaştır ve görselleştir"""
        logger.info("Model karşılaştırması başlıyor...")
        
        if not self.results:
            logger.warning("Karşılaştırılacak sonuç yok.")
            return None
        
        # Karşılaştırma verilerini oluştur
        comparison_data = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'Avg Confidence': [],
            'Training Time (s)': [],
            'Inference Time (ms)': []
        }
        
        for model_name, result in self.results.items():
            if 'accuracy' in result:
                comparison_data['Model'].append(model_name)
                comparison_data['Accuracy'].append(result.get('accuracy', 0))
                comparison_data['Precision'].append(result.get('precision', 0))
                comparison_data['Recall'].append(result.get('recall', 0))
                comparison_data['F1 Score'].append(result.get('f1_score', 0))
                comparison_data['Avg Confidence'].append(result.get('avg_confidence', 0))
                comparison_data['Training Time (s)'].append(result.get('training_time', 0))
                comparison_data['Inference Time (ms)'].append(result.get('inference_time', 0) * 1000)  # saniyeden milisaniyeye çevir
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sonuçları kaydet
        comparison_df.to_csv('models/model_comparison_results.csv', index=False)
        logger.info("Karşılaştırma sonuçları kaydedildi: models/model_comparison_results.csv")
        
        # Görselleştirme
        try:
            self._create_comparison_charts(comparison_df)
            logger.info("Karşılaştırma grafikleri oluşturuldu.")
        except Exception as e:
            logger.error(f"Grafik oluşturma hatası: {str(e)}")
        
        return comparison_df
    
    def _create_comparison_charts(self, df):
        """Karşılaştırma grafikleri oluştur"""
        plt.figure(figsize=(15, 15))
        
        # Doğruluk, Kesinlik, Duyarlılık ve F1 Skoru grafiği
        plt.subplot(3, 2, 1)
        metrics_df = df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].melt(
            id_vars=['Model'], var_name='Metric', value_name='Value'
        )
        sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df)
        plt.title('Model Performans Metrikleri')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Metrik')
        
        # Eğitim Süresi grafiği
        plt.subplot(3, 2, 2)
        sns.barplot(x='Model', y='Training Time (s)', data=df)
        plt.title('Eğitim Süresi (saniye)')
        plt.xticks(rotation=45)
        
        # Çıkarım Süresi grafiği
        plt.subplot(3, 2, 3)
        sns.barplot(x='Model', y='Inference Time (ms)', data=df)
        plt.title('Çıkarım Süresi (milisaniye)')
        plt.xticks(rotation=45)
        
        # Ortalama Güven grafiği
        plt.subplot(3, 2, 4)
        sns.barplot(x='Model', y='Avg Confidence', data=df)
        plt.title('Ortalama Güven Değeri')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Radar Chart - Tüm metrikleri birlikte göster
        plt.subplot(3, 2, 5)
        self._create_radar_chart(df)
        
        # Performans/Zaman Grafiği (Accuracy vs. Inference Time)
        plt.subplot(3, 2, 6)
        sns.scatterplot(x='Inference Time (ms)', y='Accuracy', hue='Model', size='Training Time (s)', 
                        sizes=(50, 400), data=df)
        plt.title('Doğruluk vs. Çıkarım Süresi')
        plt.xlabel('Çıkarım Süresi (ms)')
        plt.ylabel('Doğruluk')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison_charts.png', dpi=300, bbox_inches='tight')
        
        # Ayrıca interaktif bir HTML raporu oluştur
        self._create_html_report(df)
    
    def _create_radar_chart(self, df):
        """Radar chart oluştur"""
        # Kategoriler
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Avg Confidence']
        N = len(categories)
        
        # Açılar
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Çemberi tamamla
        
        # Radar chart
        ax = plt.subplot(3, 2, 5, polar=True)
        
        # Her model için
        for i, model in enumerate(df['Model']):
            values = df.loc[df['Model'] == model, categories].values.flatten().tolist()
            values += values[:1]  # Çemberi tamamla
            
            # Çiz
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Kategorileri ekle
        plt.xticks(angles[:-1], categories)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
        plt.ylim(0, 1)
        
        # Başlık ve lejant
        plt.title('Model Performans Karşılaştırması (Radar Chart)')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    def _create_html_report(self, df):
        """İnteraktif HTML raporu oluştur"""
        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Karşılaştırma Raporu</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .chart-container { margin-top: 30px; }
                    .model-section { margin-bottom: 40px; padding: 20px; border: 1px solid #eee; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Yapay Zeka Modelleri Karşılaştırma Raporu</h1>
                <p>Oluşturulma Tarihi: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                
                <h2>Performans Metrikleri Karşılaştırması</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Doğruluk (Accuracy)</th>
                        <th>Kesinlik (Precision)</th>
                        <th>Duyarlılık (Recall)</th>
                        <th>F1 Skoru</th>
                        <th>Ortalama Güven</th>
                        <th>Eğitim Süresi (s)</th>
                        <th>Çıkarım Süresi (ms)</th>
                    </tr>
            """
            
            # Tablo satırlarını ekle
            for _, row in df.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['Model']}</td>
                        <td>{row['Accuracy']:.4f}</td>
                        <td>{row['Precision']:.4f}</td>
                        <td>{row['Recall']:.4f}</td>
                        <td>{row['F1 Score']:.4f}</td>
                        <td>{row['Avg Confidence']:.4f}</td>
                        <td>{row['Training Time (s)']:.2f}</td>
                        <td>{row['Inference Time (ms)']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <div class="chart-container">
                    <h2>Karşılaştırma Grafikleri</h2>
                    <img src="../models/model_comparison_charts.png" alt="Model Comparison Charts" style="width:100%; max-width:1000px;">
                </div>
                
                <h2>Model Detayları</h2>
            """
            
            # Her model için detay bölümü
            for model_name in df['Model']:
                html_content += f"""
                <div class="model-section">
                    <h3>{model_name}</h3>
                    <p>Confusion Matrix:</p>
                    <img src="../models/{model_name.lower().replace(' ', '_')}_confusion_matrix.png" alt="{model_name} Confusion Matrix" style="width:100%; max-width:600px;">
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # HTML dosyasını kaydet
            with open('models/model_comparison_report.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info("HTML raporu oluşturuldu: models/model_comparison_report.html")
        except Exception as e:
            logger.error(f"HTML raporu oluşturma hatası: {str(e)}")
    
    def get_best_model(self, metric='accuracy'):
        """En iyi modeli belirli bir metriğe göre döndür"""
        if not self.results:
            logger.warning("Karşılaştırılacak sonuç yok.")
            return None
        
        best_score = -1
        best_model = None
        
        for model_name, result in self.results.items():
            if metric in result and result[metric] > best_score:
                best_score = result[metric]
                best_model = result['model']
        
        if best_model is None:
            logger.warning(f"En iyi model bulunamadı ({metric} metriğine göre)")
        else:
            logger.info(f"En iyi model ({metric} metriğine göre): {type(best_model).__name__}")
        
        return best_model
