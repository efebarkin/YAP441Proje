<template>
  <div class="home">
    <section class="hero">
      <div class="hero-content">
        <h1 class="hero-title">Yapay Zeka ile Tatil Planlamanın Geleceği</h1>
        <p class="hero-subtitle">TatilAI, sizin tercihlerinize göre en uygun tatil destinasyonlarını bulan yapay zeka destekli bir öneri sistemidir.</p>
        <div class="animation-container">
          <div class="vacation-animation">
            🏖️ 🏔️ 🏙️ 🏕️
          </div>
        </div>
      </div>
    </section>

    <section class="form-section">
      <div class="form-container">
        <div class="form-header">
          <div class="form-icon">
            <i class="fas fa-umbrella-beach"></i>
          </div>
          <h2 class="form-title">Tatil Önerilerinizi Alın</h2>
    </div>
        <p class="form-description">Tercihlerinizi belirtin, size en uygun tatil destinasyonlarını önerelim.</p>
        
        <div v-if="error" class="error-message">
          {{ error }}
        </div>
        
        <form @submit.prevent="submitForm" class="preference-form">
            <div class="form-group">
              <label for="budget">Bütçe (TL)</label>
    <div class="input-with-icon">
              <span class="input-icon">₺</span>
              <input 
                type="number" 
                id="budget" 
                v-model="formData.budget" 
                min="1000" 
                required
                class="form-control"
              >
            </div>
            <div class="range-display">
              <span>{{ formData.budget.toLocaleString() }} TL</span>
            </div>
              <input
                type="range"
              v-model.number="formData.budget" 
                min="1000"
                max="50000"
                step="1000"
              class="range-slider"
            >
            </div>

            <div class="form-group">
              <label for="duration">Süre (Gün)</label>
    <div class="input-with-icon">
              <input
                type="number" 
                id="duration"
                v-model="formData.duration" 
                min="1"
                max="30"
                required
                class="form-control"
              >
              <span class="input-icon">gün</span>
            </div>
            <div class="range-display">
              <span>{{ formData.duration }} gün</span>
            </div>
            <input 
              type="range" 
              v-model.number="formData.duration" 
              min="1" 
              max="30" 
              step="1" 
              class="range-slider"
            >
            </div>

            <div class="form-group">
              <label for="season">Sezon</label>
    <select 
              id="season" 
              v-model="formData.season" 
              required
              class="form-control"
            >
              <option value="" disabled>Seçiniz...</option>
              <option v-for="season in seasons" :key="season" :value="season">
                {{ season }}
              </option>
              </select>
            </div>

            <div class="form-group">
            <label for="preferred_activity">Tercih Edilen Aktivite</label>
            <select 
              id="preferred_activity" 
              v-model="formData.preferred_activity" 
              required
              class="form-control"
            >
              <option value="" disabled>Seçiniz...</option>
              <option v-for="activity in activities" :key="activity" :value="activity">
                {{ activity }}
              </option>
            </select>
          </div>
          
          <div class="form-group algorithm-selection">
            <label for="algorithm">Öneri Algoritması</label>
            <div class="custom-select">
              <select id="algorithm" v-model="formData.algorithm" class="form-control">
                <option v-for="algo in algorithms" :key="algo.id" :value="algo.id">
                  {{ algo.name }}
                </option>
              </select>
              <div class="select-arrow">
                <i class="fas fa-chevron-down"></i>
              </div>
            </div>
            <div class="algorithm-description" v-if="selectedAlgorithmDescription">
              <div class="algorithm-info-icon">
                <i class="fas fa-info-circle"></i>
              </div>
              <p>{{ selectedAlgorithmDescription }}</p>
            </div>
            </div>

          <div class="form-actions">
            <button type="submit" class="submit-button" :disabled="loading">
              <span v-if="!loading">Öneriler Al</span>
              <span v-else class="loading-spinner"></span>
              </button>
            </div>
          </form>
        </div>
    </section>
    
    <section class="features-section">
      <h2 class="section-title">Neden TatilAI?</h2>
      <div class="features-grid">
        <div class="feature-card">
          <div class="feature-icon">🧠</div>
          <h3>Yapay Zeka Destekli</h3>
          <p>Gelişmiş makine öğrenmesi algoritmaları ile kişiselleştirilmiş öneriler</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">💰</div>
          <h3>Bütçe Dostu</h3>
          <p>Bütçenize uygun tatil seçenekleri ile para tasarrufu sağlayın</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🎯</div>
          <h3>Kişiselleştirilmiş</h3>
          <p>Tercihlerinize ve ilgi alanlarınıza göre özelleştirilmiş öneriler</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">⚡</div>
          <h3>Hızlı ve Kolay</h3>
          <p>Saniyeler içinde en uygun tatil önerilerini alın</p>
        </div>
      </div>
    </section>
  </div>
</template>

<script>
import axios from 'axios'
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'

export default {
  name: 'HomeView',
  setup() {
    const router = useRouter()
    const loading = ref(false)
    const error = ref(null)
    const seasons = ref([])
    const activities = ref([])
    const algorithms = ref([])
    
    const formData = reactive({
      budget: 10000,
      duration: 7,
      season: '',
      preferred_activity: '',
      algorithm: 'decision_tree'
    })
    
    const algorithmDescriptions = {
      'decision_tree': 'Karar ağacı algoritması, kullanıcı tercihlerine göre en uygun tatil destinasyonlarını belirler. Bütçe, sezon ve aktivite tercihlerinize göre en uygun seçenekleri sunar.',
      'knn': 'K-En Yakın Komşu algoritması, benzer tercihlere sahip kullanıcıların beğendiği destinasyonları analiz ederek size özel öneriler sunar.',
      'iterative_deepening': 'Yinelemeli Derinleştirme algoritması, tüm olası tatil seçeneklerini sistematik olarak değerlendirerek size en uygun destinasyonları bulur.'
    }

    const selectedAlgorithmDescription = computed(() => {
      return algorithmDescriptions[formData.algorithm] || ''
    })

    const fetchDestinationData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/destinations')
        seasons.value = response.data.seasons
        activities.value = response.data.activities
      } catch (err) {
        console.error('Destinasyon bilgileri alınamadı:', err)
        error.value = 'Destinasyon bilgileri yüklenirken bir hata oluştu. Lütfen daha sonra tekrar deneyin.'
      }
    }

    const fetchAlgorithms = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/algorithms')
        algorithms.value = response.data.algorithms
      } catch (err) {
        console.error('Algoritma bilgileri alınamadı:', err)
        error.value = 'Algoritma bilgileri yüklenirken bir hata oluştu. Lütfen daha sonra tekrar deneyin.'
      }
    }
    
    const submitForm = async () => {
      loading.value = true
      error.value = null
      
      try {
        const requestData = {
          budget: formData.budget,
          duration: formData.duration,
          season: formData.season,
          preferred_activity: formData.preferred_activity,
          selected_algorithm: formData.algorithm
        }
        
        const response = await axios.post('http://localhost:5000/recommendation', requestData)

        // Önerileri store'a kaydet ve sonuç sayfasına yönlendir
        localStorage.setItem('recommendations', JSON.stringify(response.data.recommendations))
        localStorage.removeItem('recommendations_by_algorithm') // Artık kullanılmıyor
        router.push('/recommendations')
      } catch (err) {
        console.error('Form gönderilirken hata:', err)
        if (err.response && err.response.data && err.response.data.error) {
          error.value = err.response.data.error
        } else {
          error.value = 'Öneriler alınırken bir hata oluştu. Lütfen daha sonra tekrar deneyin.'
        }
      } finally {
        loading.value = false
      }
    }
    
    onMounted(() => {
      fetchDestinationData()
      fetchAlgorithms()
    })

    return {
      formData,
      loading,
      error,
      seasons,
      activities,
      algorithms,
      selectedAlgorithmDescription,
      submitForm
    }
  }
}
</script>

<style scoped>
.home {
  display: flex;
  flex-direction: column;
  gap: 3rem;
}

.hero {
  text-align: center;
  padding: 2rem 1rem;
  background: linear-gradient(135deg, rgba(67, 97, 238, 0.1), rgba(76, 201, 240, 0.1));
  border-radius: var(--border-radius);
  margin-bottom: 1rem;
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  display: inline-block;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: var(--light-text);
  max-width: 800px;
  margin: 0 auto;
}

.animation-container {
  margin-top: 1.5rem;
  overflow: hidden;
}

.vacation-animation {
  font-size: 2.5rem;
  animation: slide 10s linear infinite;
  white-space: nowrap;
  display: inline-block;
}

@keyframes slide {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.form-section {
  padding: 2rem 1rem;
}

.form-container {
  max-width: 800px;
  margin: 0 auto;
  background-color: var(--card-bg);
  border-radius: var(--border-radius);
  padding: 2rem;
  box-shadow: var(--card-shadow);
}

.form-header {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
}

.form-icon {
  font-size: 2rem;
  margin-right: 1rem;
}

.form-title {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: var(--primary-color);
}

.form-description {
  font-size: 1.1rem;
  color: var(--light-text);
  margin-bottom: 2rem;
}

.preference-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--dark-text);
  font-size: 1rem;
}

.form-control {
  width: 100%;
  padding: 0.8rem 1rem;
  border: 2px solid #e0e0e0;
  border-radius: var(--border-radius);
  font-size: 1rem;
  transition: all 0.3s ease;
  background-color: #fff;
  color: var(--dark-text);
}

.form-control:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15);
  outline: none;
}

.input-with-icon {
  position: relative;
  display: flex;
  align-items: center;
}

.input-icon {
  position: absolute;
  right: 1rem;
  color: var(--light-text);
  font-weight: 500;
}

.input-with-icon .input-icon:first-child {
  left: 1rem;
  right: auto;
}

.input-with-icon .form-control {
  padding-left: 2.5rem;
}

.input-with-icon .form-control + .input-icon {
  right: 1rem;
}

.range-slider {
  -webkit-appearance: none;
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: #e0e0e0;
  outline: none;
  margin: 1rem 0;
  transition: all 0.3s ease;
}

.range-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--primary-color);
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.range-slider::-webkit-slider-thumb:hover {
  transform: scale(1.1);
  background: var(--accent-color);
}

.range-slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--primary-color);
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  border: none;
}

.range-slider::-moz-range-thumb:hover {
  transform: scale(1.1);
  background: var(--accent-color);
}

.range-display {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.5rem;
  font-weight: 600;
  color: var(--primary-color);
}

.custom-select {
  position: relative;
}

.select-arrow {
  position: absolute;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--light-text);
  pointer-events: none;
}

.algorithm-selection {
  margin-top: 2rem;
}

.algorithm-description {
  margin-top: 0.8rem;
  padding: 1rem;
  background-color: rgba(76, 201, 240, 0.1);
  border-radius: var(--border-radius);
  font-size: 0.9rem;
  color: var(--light-text);
  position: relative;
  line-height: 1.5;
}

.algorithm-info-icon {
  position: absolute;
  top: 1rem;
  left: 1rem;
  color: var(--accent-color);
  font-size: 1.2rem;
}

.form-actions {
  margin-top: 2rem;
  display: flex;
  justify-content: center;
}

.submit-button {
  padding: 1rem 2.5rem;
  background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
  color: white;
  border: none;
  border-radius: var(--border-radius);
  font-weight: 600;
  font-size: 1.1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 200px;
}

.submit-button:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(67, 97, 238, 0.4);
}

.submit-button:active {
  transform: translateY(1px);
}

.submit-button:disabled {
  background: #cccccc;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}

.loading-spinner {
  display: inline-block;
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-message {
  background-color: #ffebee;
  color: #d32f2f;
  padding: 1rem;
  border-radius: var(--border-radius);
  margin-bottom: 1.5rem;
  font-weight: 500;
  display: flex;
  align-items: center;
}

.error-message:before {
  content: "⚠️";
  margin-right: 0.5rem;
  font-size: 1.2rem;
}

.features-section {
  padding: 2rem 1rem;
  max-width: 1200px;
  margin: 0 auto;
}

.section-title {
  font-size: 2rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 2rem;
  color: var(--text-color);
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

.feature-card {
  background-color: var(--card-bg);
  border-radius: var(--border-radius);
  padding: 2rem;
  box-shadow: var(--card-shadow);
  text-align: center;
  transition: transform 0.3s, box-shadow 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.feature-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

.feature-card h3 {
  font-size: 1.3rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
  color: var(--text-color);
}

.feature-card p {
  font-size: 0.95rem;
  color: var(--light-text);
  line-height: 1.5;
}

@media (max-width: 768px) {
  .hero-title {
    font-size: 2rem;
  }
  
  .hero-subtitle {
    font-size: 1rem;
  }
  
  .form-container {
    padding: 1.5rem;
  }
  
  .form-title {
    font-size: 1.5rem;
  }
  
  .form-description {
    font-size: 1rem;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
}
</style>
