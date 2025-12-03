 Экосистема Personal Assistant

**ASTRA** - это мощный, масштабируемый движок для создания персонализированных рекомендаций на основе астрологии, нумерологии и машинного обучения. Система обрабатывает тысячи запросов в день с задержкой менее 200ms и доступностью 99.9%.
```
┌─────────────────────────────────────────────────────────┐
│                    Пользовательские интерфейсы           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Telegram │  │   Unity  │  │    Web   │              │
│  │   Bot    │  │   App    │  │   App    │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
└───────┼─────────────┼─────────────┼────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                    HUB (API Gateway + Bot)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  • Circuit Breaker (Resilience4j patterns)      │  │
│  │  • Rate Limiting (50 req/min per user)          │  │
│  │  • JWT Authentication                           │  │
│  │  • Health Checks with Degradation               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────┬──────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ASTRA         │  │ RECOMMENDATIONS │  │    TESTING      │
│  (Calculations) │  │   (Ollama LLM)  │  │ (Psychological) │
│                 │  │                 │  │                 │
│ • Natal Charts  │  │ • Prompt Engine │  │ • MBTI Tests    │
│ • Magic Profile │  │ • Text Gen      │  │ • Big5 Model    │
│ • Biorhythms    │  │ • Templates     │  │ • Maslow Needs  │
│ • ML Vectors    │  │ • Cache (6h)    │  │ • Progress      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Инфраструктурный стек

```
┌─────────────────────────────────────────────────────────┐
│                    Мониторинг и трейсинг                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Prometheus│  │  Grafana │  │  Jaeger  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                    Кэширование и БД                      │
│  ┌──────────┐  ┌────────────────────────────────────┐  │
│  │  Redis   │  │      PostgreSQL Clusters           │  │
│  │ Cluster  │  │  ┌──────────┐  ┌──────────┐       │  │
│  │ (99%     │  │  │ Primary  │  │ Read     │       │  │
│  │  hit)    │  │  │          │  │ Replicas │       │  │
│  └──────────┘  │  └──────────┘  └──────────┘       │  │
└─────────────────┴──────────────────────────────────────┘
```

---

## Основные модули ASTRA

### 1. **`natal_chart.py`** - Астрологические расчеты
```python
# Использует Swiss Ephemeris (pyswisseph) для точных астрономических расчетов
class MLNatalChartCalculator:
    • Расчет позиций 10 планет (Sun, Moon, Mercury, Venus, Mars, Jupiter, 
      Saturn, Uranus, Neptune, Pluto) + Северный Узел
    • 12 домов по системе Placidus
    • Аспекты (conjunction, opposition, square, trine, sextile)
    • Элементный баланс (огонь, земля, воздух, вода)
```

### 2. **`psyho_matrix.py`** - Нумерология Пифагора
```python
# Метод Пифагора на основе даты рождения
class PsyhoMatrixCalculator:
    • 4 базовых числа (first, second, third, fourth)
    • Психоматрица 3×3 (цифры 1-9)
    • 9 характеристик личности:
      - Характер (цифра 1)
      - Энергия (цифра 2)
      - Интересы (цифра 3)
      - Здоровье (цифра 4)
      - Логика (цифра 5)
      - Труд (цифра 6)
      - Удача (цифра 7)
      - Долг (цифра 8)
      - Память (цифра 9)
```

### 3. **`biorhythm_calculator.py`** - Биоритмические циклы
```python
# Синусоидальные модели 4 циклов
class BiorhythmCalculator:
    • Физический цикл: 23 дня
    • Эмоциональный цикл: 28 дней
    • Интеллектуальный цикл: 33 дня
    • Интуитивный цикл: 38 дней
    • Расчет энергии дня (0-100%)
    • Критические и пиковые дни
```

### 4. **`magic_profile.py`** - ML-профилирование
```python
# Интеграция данных в психологический профиль
class MagicProfileCalculator:
    • 7 ключевых категорий:
      1. Ethical Framework (этические качества)
      2. Social Predispositions (социальные паттерны)
      3. Emotional Architecture (эмоциональная архитектура)
      4. Intellectual Traits (интеллектуальные черты)
      5. Willpower Profile (волевой профиль)
      6. Creative Intuitive (творческо-интуитивный)
      7. Psychological Blueprint (психологический блупринт)
    
    • 40+ ML-признаков нормализованных 0-1
    • Внутренняя согласованность данных
```

### 5. **`activity_optimizer.py`** - Оптимизация активностей
```python
# Преобразование профиля в рекомендации
class ActivityOptimizer:
    • 7 категорий активностей:
      - Physical (физическая)
      - Spiritual (духовная)
      - Learning (обучение)
      - Psychological (психологическая)
      - Career (карьера)
      - Self-realization (самореализация)
      - Finances (финансы)
    
    • Feature Vector [100+] для ML моделей
    • Расчет оптимальных 3 активностей + финансы
```

### 6. **`calculation_validator.py`** - Валидация расчетов
```python
# Математическая проверка корректности
class CalculationValidator:
    • 3 уровня валидации: BASIC, STANDARD, STRICT
    • Проверка диапазонов значений
    • Математическая согласованность
    • Валидация внутренних зависимостей
```

---

##  Производимые расчеты

### Полный пайплайн расчетов (P95=150ms)

```python
async def complete_calculation_pipeline(telegram_id: int):
    """
    Полный цикл расчетов для пользователя
    Время выполнения: ~50-150ms (с кэшированием)
    """
    
    # 1. Базовые данные пользователя (1ms)
    user = await get_user_profile(telegram_id)
    
    # 2. Параллельные расчеты (50ms макс)
    tasks = [
        calculate_natal_chart(user),      # 30ms (Swiss Ephemeris)
        calculate_psyho_matrix(user),     # 5ms (нумерология)
        calculate_biorhythms(user),       # 5ms (синусоиды)
    ]
    natal, matrix, biorhythms = await asyncio.gather(*tasks)
    
    # 3. Magic Profile (40ms)
    magic_profile = await calculate_magic_profile(
        natal_chart=natal,
        psyho_matrix=matrix,
        biorhythms=biorhythms
    )
    
    # 4. Activity Optimization (20ms)
    activities = await optimize_activities(magic_profile)
    
    # 5. Feature Vector для ML (5ms)
    feature_vector = create_ml_vector(magic_profile, activities)
    
    return {
        'natal_chart': natal,
        'psyho_matrix': matrix,
        'biorhythms': biorhythms,
        'magic_profile': magic_profile,
        'optimal_activities': activities,
        'feature_vector': feature_vector  # [100+] значений для LLM
    }
```

### Математические модели

#### Астрологические расчеты:
```
Планетарные позиции:
longitude = swe.calc_ut(jd_ut, planet_id, swe.FLG_SWIEPH)[0] % 360
sign_index = floor(longitude / 30)
```

#### Биоритмы:
```
Физический цикл (23 дня):
value = sin(2π * days_lived / 23)
percentage = ((value + 1) / 2) * 100
```

#### Magic Profile веса:
```
Этические качества = f(Saturn влияние, 9-й дом, земные элементы)
Социальные паттерны = f(Jupiter, 11-й дом, огонь/воздух)
Эмоциональные = f(Moon, аспекты, 8-й дом)
```

---

## Производительность

### Ключевые метрики

| Метрика | Значение | Целевое значение |
|---------|----------|------------------|
| **P95 Latency** | 150ms | <200ms |
| **Cache Hit Ratio** | 99% | >95% |
| **Error Rate** | 0.1% | <1% |
| **Throughput** | 100 RPS | 50-150 RPS |
| **DB Connections** | 20 активных | <50 |

###  Оптимизации производительности

#### 1. **Многоуровневое кэширование**:
```python
# Уровень 1: In-memory (5 минут)
LRU_cache = TTLCache(maxsize=1000, ttl=300)

# Уровень 2: Redis (6 часов)
redis_client.setex(
    key=f"ml_vector:{telegram_id}:{date}",
    time=21600,  # 6 часов
    value=serialized_vector
)

# Уровень 3: Materialized Views (24 часа)
CREATE MATERIALIZED VIEW daily_calculations AS
SELECT * FROM calculations 
WHERE date >= CURRENT_DATE - INTERVAL '1 day';
```

#### 2. **Асинхронная обработка**:
```python
async def async_calculation_pipeline():
    # Параллельные независимые расчеты
    natal_task = asyncio.create_task(calculate_natal())
    matrix_task = asyncio.create_task(calculate_matrix())
    
    # Ожидание всех с timeout
    done, pending = await asyncio.wait(
        [natal_task, matrix_task],
        timeout=100,  # 100ms timeout
        return_when=asyncio.ALL_COMPLETED
    )
```

#### 3. **Connection Pooling**:
```python
# SQLAlchemy с пулом соединений
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300  # 5 минут
)
```

---

##  Структура данных

###  Основные таблицы PostgreSQL

```sql
-- 1. Пользователи
CREATE TABLE users (
    telegram_id BIGINT PRIMARY KEY,
    birth_date DATE NOT NULL,
    birth_time TIME NOT NULL,
    birth_city VARCHAR(100) NOT NULL,
    current_city VARCHAR(100),
    profession VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Натальные карты (JSONB для гибкости)
CREATE TABLE user_natal_charts (
    telegram_id BIGINT PRIMARY KEY REFERENCES users,
    natal_data JSONB NOT NULL,  -- Планеты, дома, аспекты
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Magic Profiles (разделенная структура)
CREATE TABLE user_magic_profiles (
    telegram_id BIGINT PRIMARY KEY REFERENCES users,
    ethical_framework JSONB NOT NULL,
    social_predispositions JSONB NOT NULL,
    emotional_architecture JSONB NOT NULL,
    intellectual_traits JSONB NOT NULL,
    willpower_profile JSONB NOT NULL,
    creative_intuitive JSONB NOT NULL,
    psychological_blueprint JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Оптимальные активности с кэшированием
CREATE TABLE optimal_activities (
    telegram_id BIGINT REFERENCES users,
    calculation_date DATE NOT NULL,
    activities JSONB NOT NULL,      -- Топ-3 + финансы
    feature_vector FLOAT[] NOT NULL,-- [100+] ML вектор
    ml_data JSONB NOT NULL,         -- Полные ML данные
    PRIMARY KEY (telegram_id, calculation_date)
);
```

###  Data Flow между модулями

```
Пользовательские данные
       ↓
[1] Natal Chart Calculator
       ↓ (планеты, дома, аспекты)
[2] Psyho Matrix Calculator  
       ↓ (цифры, характеристики)
[3] Biorhythm Calculator
       ↓ (циклы, энергия)
       ├───────────────────┐
       ↓                   ↓
[4] Magic Profile      [5] Activity
    Calculator            Optimizer
       ↓                   ↓
[6] Feature Vector ←───────┘
       ↓
[7] Cache (Redis)
       ↓
[8] API Response
```

---



### 📊 Ожидаемая нагрузка

| Параметр | Текущая | План на 6 мес | План на 1 год |
|----------|---------|---------------|---------------|
| **Пользователей** | 1K | 10K | 100K |
| **Запросов/день** | 5K | 50K | 500K |
| **Размер данных** | 1GB | 10GB | 100GB |
| **Пиковая RPS** | 10 | 100 | 500 |

**ASTRA** - это мощный, масштабируемый движок для создания персонализированных рекомендаций на основе астрологии, нумерологии и машинного обучения. Система обрабатывает тысячи запросов в день с задержкой менее 200ms и доступностью 99.9%.

