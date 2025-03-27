import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_vacation_data(n_samples=10000):
    np.random.seed(42)
    
    # Destinasyon bilgileri - Genişletilmiş ve iyileştirilmiş
    destinations = {
        'Antalya': {
            'min_hotel_price': 1200,
            'max_hotel_price': 4500,
            'min_flight_price': 1200,
            'max_flight_price': 3500,
            'peak_seasons': ['Yaz', 'İlkbahar'],
            'off_seasons': ['Kış', 'Sonbahar'],
            'activities': {
                'Plaj': ['Yüzme', 'Güneşlenme', 'Su Sporları'],
                'Kültür': ['Müze Ziyareti', 'Tarihi Alan Gezisi', 'Yerel Mutfak'],
                'Doğa': ['Doğa Yürüyüşü', 'Kanyon Gezisi', 'Şelale Turu']
            },
            'season_activity_compatibility': {
                'Yaz': ['Plaj', 'Kültür', 'Doğa'],
                'İlkbahar': ['Plaj', 'Kültür', 'Doğa'],
                'Sonbahar': ['Kültür', 'Doğa'],
                'Kış': ['Kültür', 'Doğa']
            },
            'avg_satisfaction': 4.5
        },
        'Bodrum': {
            'min_hotel_price': 1500,
            'max_hotel_price': 6000,
            'min_flight_price': 1300,
            'max_flight_price': 3800,
            'peak_seasons': ['Yaz', 'İlkbahar'],
            'off_seasons': ['Kış', 'Sonbahar'],
            'activities': {
                'Plaj': ['Yüzme', 'Tekne Turu', 'Dalış'],
                'Kültür': ['Kale Ziyareti', 'Antik Kent Turu', 'Yerel Mutfak']
            },
            'season_activity_compatibility': {
                'Yaz': ['Plaj', 'Kültür'],
                'İlkbahar': ['Plaj', 'Kültür'],
                'Sonbahar': ['Kültür', 'Plaj'],
                'Kış': ['Kültür']
            },
            'avg_satisfaction': 4.4
        },
        'Uludağ': {
            'min_hotel_price': 1800,
            'max_hotel_price': 5200,
            'min_flight_price': 900,
            'max_flight_price': 2800,
            'peak_seasons': ['Kış'],
            'off_seasons': ['Yaz', 'İlkbahar', 'Sonbahar'],
            'activities': {
                'Kayak': ['Kayak', 'Snowboard', 'Kızak'],
                'Doğa': ['Dağ Yürüyüşü', 'Manzara Seyri', 'Kamp']
            },
            'season_activity_compatibility': {
                'Kış': ['Kayak', 'Doğa'],
                'Sonbahar': ['Doğa'],
                'İlkbahar': ['Doğa'],
                'Yaz': ['Doğa']
            },
            'avg_satisfaction': 4.3
        },
        'Kapadokya': {
            'min_hotel_price': 1100,
            'max_hotel_price': 3800,
            'min_flight_price': 1000,
            'max_flight_price': 3000,
            'peak_seasons': ['İlkbahar', 'Sonbahar'],
            'off_seasons': ['Yaz', 'Kış'],
            'activities': {
                'Kültür': ['Yeraltı Şehirleri', 'Peri Bacaları', 'Müze Ziyareti'],
                'Doğa': ['Balon Turu', 'Vadi Yürüyüşü', 'Fotoğrafçılık']
            },
            'season_activity_compatibility': {
                'İlkbahar': ['Kültür', 'Doğa'],
                'Sonbahar': ['Kültür', 'Doğa'],
                'Yaz': ['Kültür', 'Doğa'],
                'Kış': ['Kültür', 'Doğa']
            },
            'avg_satisfaction': 4.7
        },
        'Sarıkamış': {
            'min_hotel_price': 900,
            'max_hotel_price': 3000,
            'min_flight_price': 1400,
            'max_flight_price': 4000,
            'peak_seasons': ['Kış'],
            'off_seasons': ['Yaz', 'İlkbahar', 'Sonbahar'],
            'activities': {
                'Kayak': ['Kayak', 'Snowboard', 'Kış Sporları'],
                'Doğa': ['Orman Yürüyüşü', 'Kış Manzarası', 'Fotoğrafçılık']
            },
            'season_activity_compatibility': {
                'Kış': ['Kayak', 'Doğa'],
                'Sonbahar': ['Doğa'],
                'İlkbahar': ['Doğa'],
                'Yaz': ['Doğa']
            },
            'avg_satisfaction': 4.2
        },
        # Yeni eklenen destinasyonlar
        'İstanbul': {
            'min_hotel_price': 1400,
            'max_hotel_price': 6500,
            'min_flight_price': 800,
            'max_flight_price': 2800,
            'peak_seasons': ['İlkbahar', 'Sonbahar'],
            'off_seasons': ['Yaz', 'Kış'],
            'activities': {
                'Kültür': ['Tarihi Yarımada', 'Müze Ziyareti', 'Boğaz Turu'],
                'Alışveriş': ['Kapalı Çarşı', 'AVM Turu', 'Sokak Alışverişi'],
                'Gastronomi': ['Yerel Mutfak', 'Sokak Lezzetleri', 'Restoran Turu']
            },
            'season_activity_compatibility': {
                'İlkbahar': ['Kültür', 'Alışveriş', 'Gastronomi'],
                'Sonbahar': ['Kültür', 'Alışveriş', 'Gastronomi'],
                'Yaz': ['Kültür', 'Alışveriş', 'Gastronomi'],
                'Kış': ['Kültür', 'Alışveriş', 'Gastronomi']
            },
            'avg_satisfaction': 4.6
        },
        'Çanakkale': {
            'min_hotel_price': 900,
            'max_hotel_price': 3200,
            'min_flight_price': 1000,
            'max_flight_price': 3200,
            'peak_seasons': ['İlkbahar', 'Yaz'],
            'off_seasons': ['Sonbahar', 'Kış'],
            'activities': {
                'Kültür': ['Tarihi Alan Gezisi', 'Şehitlik Ziyareti', 'Müze Turu'],
                'Doğa': ['Sahil Yürüyüşü', 'Kamp', 'Doğa Fotoğrafçılığı'],
                'Deniz': ['Plaj Aktiviteleri', 'Tekne Turu', 'Balık Tutma']
            },
            'season_activity_compatibility': {
                'İlkbahar': ['Kültür', 'Doğa'],
                'Yaz': ['Kültür', 'Doğa', 'Deniz'],
                'Sonbahar': ['Kültür', 'Doğa'],
                'Kış': ['Kültür']
            },
            'avg_satisfaction': 4.5
        },
        'Samsun': {
            'min_hotel_price': 800,
            'max_hotel_price': 3000,
            'min_flight_price': 1100,
            'max_flight_price': 3300,
            'peak_seasons': ['Yaz'],
            'off_seasons': ['Kış', 'İlkbahar', 'Sonbahar'],
            'activities': {
                'Kültür': ['Kurtuluş Yolu', 'Müze Ziyareti', 'Kent Turu'],
                'Doğa': ['Sahil Yürüyüşü', 'Yayla Gezisi', 'Şehir Parkları'],
                'Plaj': ['Yüzme', 'Sahil Aktiviteleri', 'Su Sporları']
            },
            'season_activity_compatibility': {
                'Yaz': ['Kültür', 'Doğa', 'Plaj'],
                'İlkbahar': ['Kültür', 'Doğa'],
                'Sonbahar': ['Kültür', 'Doğa'],
                'Kış': ['Kültür']
            },
            'avg_satisfaction': 4.3
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
        
        # Sezon için uyumlu aktivite seç
        compatible_activities = dest_info['season_activity_compatibility'][season]
        preferred_activity = np.random.choice(compatible_activities)
        
        # Detaylı aktivite seç
        detailed_activity = np.random.choice(dest_info['activities'][preferred_activity])
        
        # Süre belirle (3-14 gün arası)
        duration = np.random.randint(3, 15)
        
        # Bütçe belirle (kişi başı) - daha gerçekçi hesaplama
        if is_peak_season:
            min_budget = (dest_info['min_hotel_price'] * duration + dest_info['min_flight_price']) * 1.2
            max_budget = (dest_info['max_hotel_price'] * duration + dest_info['max_flight_price']) * 1.2
        else:
            min_budget = dest_info['min_hotel_price'] * duration + dest_info['min_flight_price']
            max_budget = dest_info['max_hotel_price'] * duration + dest_info['max_flight_price']
        
        budget = np.random.uniform(min_budget, max_budget)
        
        # Otel fiyatı hesapla - daha gerçekçi
        if is_peak_season:
            hotel_price = np.random.uniform(dest_info['min_hotel_price'] * 1.2, 
                                          dest_info['max_hotel_price'] * 1.2)
        else:
            hotel_price = np.random.uniform(dest_info['min_hotel_price'], 
                                          dest_info['max_hotel_price'])
        
        # Uçuş fiyatı hesapla - daha gerçekçi
        if is_peak_season:
            flight_cost = np.random.uniform(dest_info['min_flight_price'] * 1.2,
                                          dest_info['max_flight_price'] * 1.2)
        else:
            flight_cost = np.random.uniform(dest_info['min_flight_price'],
                                          dest_info['max_flight_price'])
        
        # Toplam maliyet hesapla - tutarlı hesaplama
        total_cost = (hotel_price * duration) + flight_cost
        
        # Bütçe-maliyet tutarlılığı kontrolü
        # Toplam maliyet bütçeyi aşıyorsa, bütçeyi makul bir şekilde artır
        if total_cost > budget * 1.1:  # %10 tolerans
            budget = total_cost * np.random.uniform(1.05, 1.2)  # Bütçeyi %5-20 artır
        
        # Değer skoru hesapla - iyileştirilmiş formül
        value_ratio = budget / total_cost
        value_score = min(5, max(1, value_ratio * 3))
        
        # Kullanıcı memnuniyeti hesapla - daha gerçekçi
        base_satisfaction = dest_info['avg_satisfaction']
        satisfaction_modifier = (value_score / 5) * 0.4  # Değer skorunun etkisi
        season_modifier = 0.3 if is_peak_season else -0.1  # Sezon etkisi
        activity_modifier = 0.3  # Aktivite uyumlu olduğundan pozitif etki
        
        user_satisfaction = min(5, max(4, base_satisfaction + satisfaction_modifier + season_modifier + activity_modifier))
        
        # Tatil önerisi belirle - iyileştirilmiş
        recommendation_score = {
            'value_weight': 0.3,
            'satisfaction_weight': 0.3,
            'season_weight': 0.2,
            'activity_weight': 0.2
        }
        
        value_component = (value_score / 5) * recommendation_score['value_weight']
        satisfaction_component = (user_satisfaction / 5) * recommendation_score['satisfaction_weight']
        season_component = (1 if is_peak_season else 0.5) * recommendation_score['season_weight']
        activity_component = 1 * recommendation_score['activity_weight']  # Aktivite uyumlu olduğundan tam puan
        
        total_score = value_component + satisfaction_component + season_component + activity_component
        
        # En uygun tatil önerisini belirle - daha akıllı öneri sistemi
        if total_score >= 0.8:
            recommended_vacation = destination
        else:
            # Alternatif destinasyon öner - uyumlu sezon ve aktivite ile
            alternative_destinations = []
            for alt_dest, alt_info in destinations.items():
                if alt_dest != destination and preferred_activity in alt_info['season_activity_compatibility'][season]:
                    # Sezon ve aktivite uyumlu ise puan hesapla
                    alt_score = 0
                    if season in alt_info['peak_seasons']:
                        alt_score += 0.5
                    if preferred_activity in alt_info['season_activity_compatibility'][season]:
                        alt_score += 0.5
                    
                    # Yüksek puanlı alternatifleri ekle
                    if alt_score >= 0.7:
                        alternative_destinations.append(alt_dest)
            
            if alternative_destinations:
                recommended_vacation = np.random.choice(alternative_destinations)
            else:
                recommended_vacation = destination
        
        data.append({
            'user_id': _,
            'destination': destination,
            'season': season,
            'preferred_activity': preferred_activity,
            'detailed_activity': detailed_activity,
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
df = generate_vacation_data(n_samples=10000)
df.to_csv('../data/synthetic_vacation_data.csv', index=False)
print("Veri üretildi ve kaydedildi!")