<template>
  <div class="recommendations-view">
    <div class="recommendations-header">
      <h1 class="page-title">Tatil Önerileriniz</h1>
      <p class="page-subtitle">Tercihlerinize göre en uygun tatil destinasyonları</p>
      
      <div class="algorithm-info">
        <div class="algorithm-badge">
          <span class="algorithm-icon"><i class="fas fa-robot"></i></span>
          <span class="algorithm-name">{{ getAlgorithmName() }}</span>
        </div>
        <p class="algorithm-description">{{ getAlgorithmDescription() }}</p>
      </div>
    </div>
    
    <div v-if="!recommendations.length" class="no-recommendations">
      <div class="animation-container">
        <div class="no-data-animation">
          Veri bulunamadı
        </div>
      </div>
      <p>Henüz öneri bulunmuyor veya öneriler yüklenemedi.</p>
      <button @click="goBack" class="back-button">
        <i class="fas fa-arrow-left back-icon"></i> Geri Dön
      </button>
    </div>
    
    <div v-else class="recommendations-container">
      <button @click="goBack" class="back-button">
        <i class="fas fa-arrow-left back-icon"></i> Geri Dön
      </button>
      
      <div class="recommendations-list">
        <h2 class="recommendations-title">Önerilen Tatil Destinasyonları</h2>
        <transition-group name="recommendation-list" tag="div" class="recommendations-grid">
          <div 
            v-for="(recommendation, index) in recommendations" 
            :key="index"
            class="recommendation-card"
          >
            <div class="destination-image" :style="getDestinationImage(recommendation.destination)">
              <h3 class="destination-name">{{ recommendation.destination }}</h3>
            </div>
            <div class="card-content">
              <div class="recommendation-details">
                <div class="detail-item">
                  <span class="detail-label">Sezon</span>
                  <span class="detail-value">{{ recommendation.season }}</span>
                </div>
                <div class="detail-item">
                  <span class="detail-label">Aktivite</span>
                  <span class="detail-value">{{ recommendation.activity }}</span>
                </div>
                <div v-if="recommendation.costs && recommendation.costs.hotel_price" class="detail-item">
                  <span class="detail-label">Otel Fiyatı</span>
                  <span class="detail-value">{{ formatPrice(recommendation.costs.hotel_price) }} TL/gün</span>
                </div>
                <div v-if="recommendation.costs && recommendation.costs.flight_cost" class="detail-item">
                  <span class="detail-label">Uçak Bileti</span>
                  <span class="detail-value">{{ formatPrice(recommendation.costs.flight_cost) }} TL</span>
                </div>
                <div v-if="recommendation.costs && recommendation.costs.total_cost" class="detail-item">
                  <span class="detail-label">Toplam Maliyet</span>
                  <span class="detail-value">{{ formatPrice(recommendation.costs.total_cost) }} TL</span>
                </div>
              </div>
              
              <div class="confidence-details">
                <div class="progress-container">
                  <div class="progress-label">Algoritma Güven Değeri</div>
                  <div class="progress-bar-container">
                    <div 
                      class="progress-bar" 
                      :style="{ width: `${(recommendation.algorithm_confidence || 0.5) * 100}%` }"
                    ></div>
                  </div>
                  <div class="progress-value">{{ Math.round((recommendation.algorithm_confidence || 0.5) * 100) }}%</div>
                </div>
              </div>
              
              <button class="book-button">Rezervasyon Yap</button>
            </div>
          </div>
        </transition-group>
      </div>
      
      <div class="back-section">
        <button @click="goBack" class="back-button">
          <i class="fas fa-arrow-left back-icon"></i> Geri Dön
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

export default {
  name: 'RecommendationsView',
  setup() {
    const router = useRouter()
    const recommendations = ref([])
    const loading = ref(true)
    const error = ref(null)
    const algorithms = ref([])
    
    const fetchAlgorithms = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/algorithms')
        algorithms.value = response.data.algorithms
      } catch (err) {
        console.error('Algoritma bilgileri alınamadı:', err)
      }
    }
    
    const getAlgorithmName = () => {
      if (!recommendations.value || recommendations.value.length === 0) {
        return 'Algoritma'
      }
      
      const rec = recommendations.value[0]
      console.log('Recommendation object:', rec);
      const algorithmId = rec.algorithm || 'decision_tree'
      console.log('Algorithm ID:', algorithmId);
      
      // Algoritma adları
      const algorithmNames = {
        'decision_tree': 'Karar Ağacı',
        'a_star': 'A* Algoritması',
        'genetic': 'Genetik Algoritma',
        'iterative_deepening': 'Iterative Deepening',
        'knn': 'K-En Yakın Komşu'
      };
      
      return algorithmNames[algorithmId] || 'Karar Ağacı';
    }
    
    const getAlgorithmDescription = () => {
      if (!recommendations.value || recommendations.value.length === 0) {
        return 'Yapay zeka destekli öneri sistemi'
      }
      
      const rec = recommendations.value[0]
      const algorithmId = rec.algorithm || 'decision_tree'
      
      // Algoritma açıklamaları
      const algorithmDescriptions = {
        'decision_tree': 'Basit karar kuralları kullanarak tahmin yapar, kolay anlaşılır ve yorumlanabilir sonuçlar üretir.',
        'a_star': 'Heuristic tabanlı arama algoritması, en uygun tatil destinasyonunu bulmak için özellik ağırlıklarını kullanır.',
        'genetic': 'Evrimsel hesaplama yaklaşımı kullanarak, tercihlerinize en uygun tatil paketlerini oluşturur.',
        'iterative_deepening': 'Derinlik sınırlı arama ile tercihlerinize en uygun destinasyonları belirler.',
        'knn': 'Benzer kullanıcıların tercihlerini analiz ederek size en uygun tatil önerilerini sunar.'
      };
      
      return algorithmDescriptions[algorithmId] || 'Basit karar kuralları kullanarak tahmin yapar';
    }
    
    const loadRecommendations = () => {
      try {
        const storedRecommendations = localStorage.getItem('recommendations')
        
        if (storedRecommendations) {
          recommendations.value = JSON.parse(storedRecommendations)
          console.log('Loaded recommendations:', recommendations.value);
          
          // Ensure algorithm property exists in each recommendation
          if (recommendations.value.length > 0 && !recommendations.value[0].algorithm) {
            // Try to get algorithm from localStorage
            const algorithm = localStorage.getItem('selectedAlgorithm')
            if (algorithm) {
              recommendations.value.forEach(rec => {
                rec.algorithm = algorithm
              })
            }
          }
        }
        
        if (!storedRecommendations) {
          router.push('/')
        }
      } catch (err) {
        console.error('Öneriler yüklenirken hata:', err)
        error.value = 'Öneriler yüklenirken bir hata oluştu.'
      } finally {
        loading.value = false
      }
    }
    
    const goBack = () => {
      router.push('/')
    }
    
    const formatPrice = (price) => {
      return new Intl.NumberFormat('tr-TR').format(price)
    }
    
    const getConfidenceClass = (confidence) => {
      if (confidence >= 0.8) return 'high'
      if (confidence >= 0.6) return 'medium'
      return 'low'
    }
    
    const getDestinationImage = (destination) => {
      // Destinasyona göre arka plan resmi seç
      let backgroundImage;
      
      // Destinasyona göre arka plan resmini belirle
      switch(destination) {
        case 'Antalya':
          backgroundImage = 'url(/img/Antalya.png)';
          break;
        case 'Bodrum':
          backgroundImage = 'url(/img/bodrum.png)';
          break;
        case 'Kapadokya':
          backgroundImage = 'url(/img/kapadokya.png)';
          break;
        case 'Uludağ':
          backgroundImage = 'url(/img/uludag.png)';
          break;
        case 'Sarıkamış':
          backgroundImage = 'url(/img/sarikamis.png)';
          break;
        case 'İstanbul':
          backgroundImage = 'url(https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)';
          break;
        case 'Çanakkale':
          backgroundImage = 'url(https://images.unsplash.com/photo-1600598439902-40408abc471d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)';
          break;
        default:
          backgroundImage = 'url(https://images.unsplash.com/photo-1517760444937-f6397edcbbcd?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)';
      }
      
      // Style objesi döndür
      return {
        backgroundImage: backgroundImage,
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      };
    };
    
    onMounted(() => {
      fetchAlgorithms()
      loadRecommendations()
    })
    
    return {
      recommendations,
      loading,
      error,
      goBack,
      getAlgorithmName,
      getAlgorithmDescription,
      formatPrice,
      getConfidenceClass,
      getDestinationImage
    }
  }
}
</script>

<style scoped>
.recommendations-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.recommendations-header {
  text-align: center;
  margin-bottom: 3rem;
  position: relative;
}

.page-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  background: linear-gradient(135deg, #4361ee, #4cc9f0);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  display: inline-block;
}

.page-subtitle {
  font-size: 1.2rem;
  color: #6c757d;
  margin-bottom: 2rem;
}

.algorithm-info {
  background: linear-gradient(135deg, rgba(67, 97, 238, 0.05), rgba(76, 201, 240, 0.05));
  border-radius: 12px;
  padding: 1.5rem;
  margin: 0 auto 2rem;
  max-width: 800px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
  position: relative;
  overflow: hidden;
}

.algorithm-info::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, #4361ee, #4cc9f0);
}

.algorithm-badge {
  display: inline-flex;
  align-items: center;
  background: linear-gradient(135deg, #4361ee, #4cc9f0);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 30px;
  margin-bottom: 1rem;
  box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3);
}

.algorithm-icon {
  margin-right: 0.5rem;
  font-size: 1.1rem;
}

.algorithm-name {
  font-weight: 600;
  font-size: 1.1rem;
}

.algorithm-description {
  color: #495057;
  line-height: 1.6;
  font-size: 1.05rem;
}

.recommendations-container {
  position: relative;
}

.back-button {
  display: inline-flex;
  align-items: center;
  padding: 0.8rem 1.5rem;
  background-color: #f8f9fa;
  color: #495057;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-bottom: 2rem;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.back-button:hover {
  background-color: #e9ecef;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.back-button:active {
  transform: translateY(0);
}

.back-icon {
  margin-right: 0.5rem;
}

.recommendations-list {
  margin-bottom: 3rem;
}

.recommendations-title {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #343a40;
  position: relative;
  display: inline-block;
}

.recommendations-title::after {
  content: '';
  position: absolute;
  bottom: -10px;
  left: 0;
  width: 50px;
  height: 4px;
  background: linear-gradient(90deg, #4361ee, #4cc9f0);
  border-radius: 2px;
}

.recommendations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
}

.recommendation-card {
  background-color: #fff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  position: relative;
}

.recommendation-card:hover {
  transform: translateY(-10px);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.12);
}

.destination-image {
  height: 200px;
  background-size: cover;
  background-position: center;
  position: relative;
}

.destination-image::before {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 70%;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent);
}

.destination-name {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  color: white;
  padding: 1.5rem 1.5rem 1rem;
  margin: 0;
  font-size: 1.8rem;
  font-weight: 700;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.card-content {
  padding: 1.5rem;
}

.recommendation-details {
  margin-bottom: 1.5rem;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.8rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid #f0f0f0;
}

.detail-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}

.detail-label {
  font-weight: 600;
  color: #6c757d;
}

.detail-value {
  color: #343a40;
  font-weight: 600;
}

.confidence-details {
  margin: 1.5rem 0;
}

.progress-container {
  margin-bottom: 1rem;
}

.progress-label {
  font-weight: 600;
  margin-bottom: 0.8rem;
  color: #495057;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.progress-bar-container {
  height: 10px;
  background-color: #e9ecef;
  border-radius: 5px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(to right, #4361ee, #4cc9f0);
  border-radius: 5px;
  transition: width 0.8s ease;
}

.progress-value {
  text-align: right;
  font-weight: 600;
  color: #495057;
  margin-top: 0.5rem;
}

.book-button {
  display: block;
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #4361ee, #4cc9f0);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1.1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3);
  text-align: center;
}

.book-button:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 15px rgba(67, 97, 238, 0.4);
}

.book-button:active {
  transform: translateY(-1px);
}

.no-recommendations {
  text-align: center;
  padding: 3rem;
  background-color: #f8f9fa;
  border-radius: 12px;
  margin: 2rem 0;
}

.no-data-animation {
  font-size: 3rem;
  margin-bottom: 1.5rem;
  color: #6c757d;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 0.6; transform: scale(0.95); }
  50% { opacity: 1; transform: scale(1); }
  100% { opacity: 0.6; transform: scale(0.95); }
}

.back-section {
  margin-top: 3rem;
  text-align: center;
}

/* Animasyonlar */
.recommendation-list-enter-active,
.recommendation-list-leave-active {
  transition: all 0.5s ease;
}

.recommendation-list-enter-from,
.recommendation-list-leave-to {
  opacity: 0;
  transform: translateY(30px);
}

.recommendation-list-move {
  transition: transform 0.5s ease;
}

@media (max-width: 768px) {
  .recommendations-grid {
    grid-template-columns: 1fr;
  }
  
  .page-title {
    font-size: 2rem;
  }
}
</style>
