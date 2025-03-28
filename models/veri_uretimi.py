import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_vacation_data(n_samples=5000):
    np.random.seed(42)
    
    # Destinasyon bilgileri
    destinations = {
        'Antalya': {
            'min_hotel_price': 800,
            'max_hotel_price': 3000,
            'min_flight_price': 800,
            'max_flight_price': 2500,
            'peak_seasons': ['Yaz'],
            'activities': ['Plaj', 'Kültür', 'Doğa'],
            'avg_satisfaction': 4.5
        },
        'Bodrum': {
            'min_hotel_price': 1000,
            'max_hotel_price': 4000,
            'min_flight_price': 900,
            'max_flight_price': 2800,
            'peak_seasons': ['Yaz'],
            'activities': ['Plaj', 'Kültür'],
            'avg_satisfaction': 4.4
        },
        'Uludağ': {
            'min_hotel_price': 1200,
            'max_hotel_price': 3500,
            'min_flight_price': 600,
            'max_flight_price': 2000,
            'peak_seasons': ['Kış'],
            'activities': ['Kayak', 'Doğa'],
            'avg_satisfaction': 4.3
        },
        'Kapadokya': {
            'min_hotel_price': 700,
            'max_hotel_price': 2500,
            'min_flight_price': 700,
            'max_flight_price': 2200,
            'peak_seasons': ['İlkbahar', 'Sonbahar'],
            'activities': ['Kültür', 'Doğa'],
            'avg_satisfaction': 4.7
        },
        'Sarıkamış': {
            'min_hotel_price': 600,
            'max_hotel_price': 2000,
            'min_flight_price': 1000,
            'max_flight_price': 3000,
            'peak_seasons': ['Kış'],
            'activities': ['Kayak', 'Doğa'],
            'avg_satisfaction': 4.2
        }
    }
    
    data = []
    
    for _ in range(n_samples):
        # Rastgele bir destinasyon seç
        destination = np.random.choice(list(destinations.keys()))
        dest_info = destinations[destination]
        
        # Sezon seç
        season = np.random.choice(['Yaz', 'Kış', 'İlkbahar', 'Sonbahar'])
        is_peak_season = season in dest_info['peak_seasons']
        
        # Aktivite seç
        preferred_activity = np.random.choice(dest_info['activities'])
        
        # Süre belirle (3-14 gün arası)
        duration = np.random.randint(3, 15)
        
        # Bütçe belirle (kişi başı)
        if is_peak_season:
            min_budget = (dest_info['min_hotel_price'] * duration + dest_info['min_flight_price']) * 1.3
            max_budget = (dest_info['max_hotel_price'] * duration + dest_info['max_flight_price']) * 1.3
        else:
            min_budget = dest_info['min_hotel_price'] * duration + dest_info['min_flight_price']
            max_budget = dest_info['max_hotel_price'] * duration + dest_info['max_flight_price']
        
        budget = np.random.uniform(min_budget, max_budget)
        
        # Otel fiyatı hesapla
        if is_peak_season:
            hotel_price = np.random.uniform(dest_info['min_hotel_price'] * 1.3, 
                                          dest_info['max_hotel_price'] * 1.3)
        else:
            hotel_price = np.random.uniform(dest_info['min_hotel_price'], 
                                          dest_info['max_hotel_price'])
        
        # Uçuş fiyatı hesapla
        if is_peak_season:
            flight_cost = np.random.uniform(dest_info['min_flight_price'] * 1.3,
                                          dest_info['max_flight_price'] * 1.3)
        else:
            flight_cost = np.random.uniform(dest_info['min_flight_price'],
                                          dest_info['max_flight_price'])
        
        # Toplam maliyet hesapla
        total_cost = (hotel_price * duration) + flight_cost
        
        # Değer skoru hesapla
        value_ratio = budget / total_cost
        value_score = min(5, max(1, value_ratio * 3))
        
        # Kullanıcı memnuniyeti hesapla
        base_satisfaction = dest_info['avg_satisfaction']
        satisfaction_modifier = (value_score / 5) * 0.5  # Değer skorunun etkisi
        season_modifier = 0.3 if is_peak_season else -0.1  # Sezon etkisi
        activity_modifier = 0.2 if preferred_activity in dest_info['activities'] else -0.2  # Aktivite uyumu
        
        user_satisfaction = min(5, max(1, base_satisfaction + satisfaction_modifier + season_modifier + activity_modifier))
        
        # Tatil önerisi belirle
        recommendation_score = {
            'value_weight': 0.3,
            'satisfaction_weight': 0.3,
            'season_weight': 0.2,
            'activity_weight': 0.2
        }
        
        value_component = (value_score / 5) * recommendation_score['value_weight']
        satisfaction_component = (user_satisfaction / 5) * recommendation_score['satisfaction_weight']
        season_component = (1 if is_peak_season else 0.5) * recommendation_score['season_weight']
        activity_component = (1 if preferred_activity in dest_info['activities'] else 0.5) * recommendation_score['activity_weight']
        
        total_score = value_component + satisfaction_component + season_component + activity_component
        
        # En uygun tatil önerisini belirle
        if total_score >= 0.8:
            recommended_vacation = destination
        else:
            # Alternatif destinasyon öner
            alternative_destinations = [d for d in destinations.keys() 
                                     if d != destination and 
                                     preferred_activity in destinations[d]['activities'] and
                                     season in destinations[d]['peak_seasons']]
            if alternative_destinations:
                recommended_vacation = np.random.choice(alternative_destinations)
            else:
                recommended_vacation = destination
        
        data.append({
            'user_id': _,
            'destination': destination,
            'season': season,
            'preferred_activity': preferred_activity,
            'duration': duration,
            'budget': round(budget, 2),
            'hotel_price_per_night': round(hotel_price, 2),
            'flight_cost': round(flight_cost, 2),
            'total_cost': round(total_cost, 2),
            'value_score': round(value_score, 2),
            'user_satisfaction': round(user_satisfaction, 2),
            'recommended_vacation': recommended_vacation
        })
    
    df = pd.DataFrame(data)
    return df

# Veri üret ve kaydet
df = generate_vacation_data(n_samples=5000)
df.to_csv('synthetic_vacation_data.csv', index=False)
print("Veri üretildi ve kaydedildi!")