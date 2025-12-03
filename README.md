Проект magic_profile является частью проекта personal_assistant состоящего из четырех проектов (с отдельными репозиториями):

astra: подготовка натальных карт, психоматриц, расчета биоритмов и рекоммендаций на один день

testing: психологическое тестирование и потребностей пользователя

assistant: на основании расчетов magic_profile, astra и testing выдает персонализированные рекоммендации на один день на базе ollama

astra: развернут локально (включая бд), взаимодействие телеграм бот 

расчеты: сбор первичной информации пользователя, расчет натальной карты, психоматрицы, биоритмов, лунных фаз, расчет данных на день.

проект magic_profile: создание и запись в базу данных натального и психоматричного профиля пользователя

структура проекта magic_profile:

magic_profile/
├── docker-compose.yml
├── requirements.txt
├── .env
│-- ephe
├── bot/                          # Только для тестирования
│   ├── config.py
│   ├── testing_handlers.py       # Вывод данных для модели
│   └── main.py
│
├── backend/
│   ├── __init__.py
│   ├── database.py               # Модели и подключение к БД
│   ├── profile_builder.py        # Основной конструктор профиля
│   ├── calculators/              # Все калькуляторы
│   │   ├── natal_calculator.py
│   │   ├── matrix_calculator.py
│   │   ├── energy_calculator.py
│   │   ├── psychology_calculator.py
│   │   └── financial_calculator.py
│   ├── optimizers/               # Оптимизаторы данных
│   │   ├── profile_optimizer.py
│   │   └── model_preparer.py
│   └── api/
│       ├── astra_integration.py  # Точка входа из Astra
│       └── assistant_api.py      # API для Assistant
│
|--- init-scripts
|    |--- 01-init-tables.sql
|
└── tests/
    ├── test_profile_builder.py
    └── test_calculators.py
    
 
    
Производимые расчеты прокта:

A. МОРАЛЬНО-ЭТИЧЕСКИЕ КАЧЕСТВА
{
  "ethical_framework": {
    "honesty_tendency": 0.7,          # Saturn + 9-й дом + земные элементы
    "discretion_level": 0.6,          # Pluto + 8-й дом + водные знаки  
    "responsibility_capacity": 0.8,   # Saturn + 10-й дом + Козерог
    "loyalty_expression": 0.5,        # Луна + 4-й дом + водные элементы
    
    "calculated_metrics": {
      "truth_priority": 0.8,          # Приоритет правды над комфортом
      "privacy_need": 0.7,            # Потребность в сохранении тайн
      "commitment_strength": 0.6,     # Сила обязательств
      "trust_building_speed": 0.4     # Скорость установления доверия
    }
  }
}

B. СОЦИАЛЬНЫЕ ПАТТЕРНЫ
{
  "social_predispositions": {
    "extroversion_level": 0.3,        # Огненные/воздушные vs земные/водные
    "empathy_capacity": 0.6,          # Луна + Нептун + водные знаки
    "conflict_approach": "analytical", # Марс + Меркурий аспекты
    "group_dynamics_skill": 0.5,      # 11-й дом + Юпитер
    
    "interaction_patterns": {
      "assertiveness": 0.7,           # Марс + 1-й дом
      "diplomacy_skill": 0.4,         # Весы + 7-й дом
      "listening_ability": 0.8,       # Луна + Рак
      "boundary_setting": 0.6         # Сатурн + Скорпион
    }
  }
}

C. ЭМОЦИОНАЛЬНЫЕ ХАРАКТЕРИСТИКИ
{
  "emotional_architecture": {
    "emotional_stability": 0.7,       # Луна + Сатурн аспекты
    "vulnerability_comfort": 0.4,     # Луна + 8-й дом
    "anger_expression": "controlled", # Марс + Сатурн
    "joy_capacity": 0.8,              # Венера + Юпитер
    
    "regulation_patterns": {
      "self_awareness": 0.6,          # Луна + Меркурий
      "impulse_control": 0.7,         # Марс + Сатурн
      "stress_resilience": 0.5,       # Сатурн + 6-й дом
      "mood_consistency": 0.8         # Луна в фиксированном знаке
    }
  }
}

D. ИНТЕЛЛЕКТУАЛЬНЫЕ ПРЕДРАСПОЛОЖЕННОСТИ
{
  "intellectual_traits": {
    "curiosity_level": 0.8,           # Меркурий + Стрелец + 9-й дом
    "skepticism_tendency": 0.6,       # Сатурн + Дева + 3-й дом
    "learning_agility": 0.7,          # Меркурий + Уран аспекты
    "knowledge_retention": 0.5,       # Луна + Сатурн
    
    "thinking_patterns": {
      "critical_thinking": 0.7,       # Меркурий + Сатурн
      "creative_synthesis": 0.8,      # Меркурий + Нептун
      "systemic_thinking": 0.6,       # Сатурн + 3-й дом
      "practical_application": 0.9    # Земные знаки + 6-й дом
    }
  }
}

E. ВОЛЕВЫЕ КАЧЕСТВА
{
  "willpower_profile": {
    "determination_strength": 0.8,    # Марс + Скорпион + 1-й дом
    "persistence_capacity": 0.7,      # Сатурн + Телец + фиксированные знаки
    "adaptability_speed": 0.4,        # Меркурий + Близнецы + мутабельные знаки
    "initiative_taking": 0.9,         # Марс + Овен + 1-й дом
    
    "execution_traits": {
      "procrastination_tendency": 0.3, # Сатурн слабый + Нептун сильный
      "follow_through_ability": 0.8,   # Сатурн + земные знаки
      "multitasking_capacity": 0.5,    # Близнецы + 3-й дом
      "focus_depth": 0.7               # Скорпион + 8-й дом
    }
  }
}

F. ТВОРЧЕСКИЕ И ИНТУИТИВНЫЕ СПОСОБНОСТИ
{
  "creative_intuitive": {
    "imagination_vividness": 0.6,     # Нептун + Рыбы + 12-й дом
    "intuition_strength": 0.8,        # Луна + Нептун + водные знаки
    "innovation_capacity": 0.7,       # Уран + Водолей + 11-й дом
    "artistic_sensitivity": 0.5,      # Венера + Нептун
    
    "inspiration_patterns": {
      "dream_utilization": 0.4,       # Луна + 12-й дом
      "symbol_interpretation": 0.7,   # Нептун + Скорпион
      "pattern_recognition": 0.9,     # Меркурий + Дева
      "cross_domain_synthesis": 0.6   # Юпитер + 9-й дом
    }
  }
}

ИНТЕГРИРОВАННАЯ СТРУКТУРА ДЛЯ МОДЕЛИ
{
  "psychological_blueprint": {
    
    # 🎭 БАЗОВЫЕ ЛИЧНОСТНЫЕ ЧЕРТЫ
    "core_personality": {
      "integrity_index": 0.7,         # Общий индекс честности/надежности
      "openness_balance": 0.6,        # Баланс открытости/скрытности
      "dependability_score": 0.8,     # Надежность и ответственность
      "authenticity_level": 0.7       # Естественность самовыражения
    },
    
    # 🤝 СОЦИАЛЬНАЯ АРХИТЕКТУРА  
    "social_architecture": {
      "trust_dynamics": {
        "trust_giving_speed": 0.4,    # Скорость доверия к другим
        "trust_earning_need": 0.8,    # Потребность в доверии от других
        "betrayal_resilience": 0.5,   # Устойчивость к предательству
        "loyalty_expression": 0.7     # Стиль проявления верности
      },
      
      "communication_ethics": {
        "transparency_preference": 0.6,  # Предпочтение прозрачности
        "diplomacy_priority": 0.4,       # Приоритет дипломатии над правдой
        "confidentiality_respect": 0.8,  # Уважение к конфиденциальности
        "directness_comfort": 0.7        # Комфорт с прямотой
      }
    },
    
    # ⚖️ МОРАЛЬНЫЕ ОРИЕНТИРЫ
    "moral_compass": {
      "rule_following_tendency": 0.6,    # Следование правилам vs гибкость
      "justice_sensitivity": 0.8,        # Чувствительность к несправедливости
      "forgiveness_capacity": 0.5,       # Способность прощать
      "consistency_importance": 0.7      # Важность последовательности
    },
    
    # 🛡️ ЗАЩИТНЫЕ МЕХАНИЗМЫ
    "defense_mechanisms": {
      "vulnerability_shielding": 0.6,    # Защита уязвимости
      "emotional_armor": 0.7,            # Эмоциональная броня
      "information_guarding": 0.8,       # Охрана информации
      "boundary_strength": 0.5           # Сила личных границ
    },
    
    # 📊 ПРАКТИЧЕСКИЕ ПРОЯВЛЕНИЯ
    "behavioral_manifestations": {
      "promise_keeping": 0.8,           # Соблюдение обещаний
      "secret_keeping_ability": 0.7,    # Способность хранить тайны
      "accountability_taking": 0.9,     # Принятие ответственности
      "authentic_expression": 0.6       # Подлинное самовыражение
    }
  }
}

Для построения доверия:
trust_strategy = f"""
Профиль доверия пользователя:
- Скорость установления доверия: {trust_speed}
- Потребность в прозрачности: {transparency_need}
- Уважение к конфиденциальности: {confidentiality_respect}

Рекомендуемая стратегия: {trust_building_approach}
"""

Для командной работы:
team_dynamics = f"""
Социальные паттерны для команды:
- Надежность: {dependability_score}
- Прямота в общении: {directness_comfort} 
- Конфликтный стиль: {conflict_approach}

Оптимальная роль: {team_role_recommendation}
"""

Для личного развития:
growth_focus = f"""
Приоритеты развития характера:
1. Усилить {strength_to_develop} через {development_method}
2. Сбалансировать {trait_to_balance} с помощью {balancing_approach}
3. Использовать {natural_trait} для компенсации {challenge_area}
"""



    
    
   
   
   
