-- 1. ОСНОВНАЯ ТАБЛИЦА ПРОФИЛЕЙ
CREATE TABLE magic_profiles (
    profile_id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE NOT NULL,
    profile_version INTEGER DEFAULT 1,
    overall_confidence_score DECIMAL(3,2), -- Общая уверенность расчетов
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ссылка на пользователя в Astra (опционально)
    CONSTRAINT fk_astra_user FOREIGN KEY(telegram_id) 
    REFERENCES astra_db.users(telegram_id) ON DELETE CASCADE
);

-- 2. ЭТИЧЕСКИЕ КАЧЕСТВА
CREATE TABLE ethical_qualities (
    profile_id INTEGER REFERENCES magic_profiles(profile_id),
    quality_type VARCHAR(50) NOT NULL,
    score DECIMAL(3,2) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (profile_id, quality_type),
    CHECK (score BETWEEN 0 AND 1),
    CHECK (confidence BETWEEN 0 AND 1)
);

-- 3. СОЦИАЛЬНЫЕ ПАТТЕРНЫ
CREATE TABLE social_patterns (
    profile_id INTEGER REFERENCES magic_profiles(profile_id),
    pattern_type VARCHAR(50) NOT NULL,
    score DECIMAL(3,2) NOT NULL,
    characteristics JSONB NOT NULL,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (profile_id, pattern_type)
);

-- 4. ЭМОЦИОНАЛЬНЫЕ ХАРАКТЕРИСТИКИ
CREATE TABLE emotional_architecture (
    profile_id INTEGER REFERENCES magic_profiles(profile_id),
    dimension VARCHAR(50) NOT NULL,
    stability_score DECIMAL(3,2),
    expression_style VARCHAR(20),
    regulation_patterns JSONB,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (profile_id, dimension)
);

-- 5. ИНТЕЛЛЕКТУАЛЬНЫЕ ПРЕДРАСПОЛОЖЕННОСТИ
CREATE TABLE intellectual_traits (
    profile_id INTEGER REFERENCES magic_profiles(profile_id),
    trait_type VARCHAR(50) NOT NULL,
    capacity_score DECIMAL(3,2),
    thinking_style VARCHAR(30),
    learning_preferences JSONB,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (profile_id, trait_type)
);

-- 6. ВОЛЕВЫЕ КАЧЕСТВА (ОБЪЕДИНЯЕМ willpower + creative)
CREATE TABLE volitional_qualities (
    profile_id INTEGER REFERENCES magic_profiles(profile_id),
    quality_category VARCHAR(30) NOT NULL, -- 'willpower', 'creativity', 'intuition'
    quality_name VARCHAR(50) NOT NULL,
    strength_score DECIMAL(3,2),
    manifestation_style VARCHAR(30),
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (profile_id, quality_category, quality_name)
);

-- 7. ИНТЕГРИРОВАННЫЙ ПСИХОЛОГИЧЕСКИЙ БЛУПРИНТ (СВОДНАЯ)
CREATE TABLE psychological_blueprint (
    profile_id INTEGER PRIMARY KEY REFERENCES magic_profiles(profile_id),
    core_personality JSONB NOT NULL,
    social_architecture JSONB NOT NULL,
    moral_compass JSONB NOT NULL,
    defense_mechanisms JSONB NOT NULL,
    behavioral_manifestations JSONB NOT NULL,
    
    -- Метрики качества данных
    data_quality_score INTEGER DEFAULT 0,
    calculation_coverage DECIMAL(3,2) DEFAULT 0,
    last_validated_at TIMESTAMPTZ,
    
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- Основные индексы
CREATE INDEX idx_profiles_telegram ON magic_profiles(telegram_id);
CREATE INDEX idx_profiles_updated ON magic_profiles(updated_at);
CREATE INDEX idx_profiles_active ON magic_profiles(is_active) WHERE is_active = true;

-- Индексы для поиска по качествам
CREATE INDEX idx_ethical_scores ON ethical_qualities(score) WHERE score > 0.7;
CREATE INDEX idx_social_patterns ON social_patterns USING gin(characteristics);
CREATE INDEX idx_emotional_stability ON emotional_architecture(stability_score);

-- Индексы для аналитики
CREATE INDEX idx_blueprint_quality ON psychological_blueprint(data_quality_score);
CREATE INDEX idx_calculated_recent ON emotional_architecture(calculated_at DESC);

