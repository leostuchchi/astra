Проект astra является частью микросервисной экосистемы personal_assistant состоящего из пяти проектов (с отдельными репозиториями и основным .yaml, для запуска всех подконтейнеров в одом Docker):

DB: Postgres, до 1000 пользователей, единая база данных для всех проектов.

структура проекта astra:

backend:
backend.config.assistant_api_config.py:
import os
from typing import Dict, Any


class AssistantAPIConfig:
    """Конфигурация API для assistant"""

    # Настройки API
    API_TITLE = "Personal Assistant API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "API для доступа к оптимальным активностям и рекомендациям"

    # Настройки сервера
    HOST = os.getenv("ASSISTANT_API_HOST", "0.0.0.0")
    PORT = int(os.getenv("ASSISTANT_API_PORT", "8000"))
    RELOAD = os.getenv("ASSISTANT_API_RELOAD", "False").lower() == "true"

    # Настройки CORS
    CORS_ORIGINS = os.getenv("ASSISTANT_API_CORS_ORIGINS", "*").split(",")

    # Настройки аутентификации
    API_KEYS = os.getenv("ASSISTANT_API_KEYS", "").split(",")

    # Настройки кэширования
    CACHE_TTL = int(os.getenv("ASSISTANT_API_CACHE_TTL", "300"))  # 5 минут

    # Настройки логирования
    LOG_LEVEL = os.getenv("ASSISTANT_API_LOG_LEVEL", "INFO")

    @classmethod
    def get_fastapi_config(cls) -> Dict[str, Any]:
        """Получение конфигурации для FastAPI"""
        return {
            "title": cls.API_TITLE,
            "description": cls.API_DESCRIPTION,
            "version": cls.API_VERSION,
            "docs_url": "/docs",
            "redoc_url": "/redoc"
        }
# Экспорт конфигурации
config = AssistantAPIConfig()
backend.activity_optimizer.py:
import logging
import asyncio
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import math
from enum import Enum
from sqlalchemy.future import select
from sqlalchemy import func

from backend.database import async_session, OptimalActivities
from backend.magic_profile import magic_profile_service

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Типы активностей для оптимизации - только для внутреннего использования"""
    PHYSICAL = "physical"
    SPIRITUAL = "spiritual"
    LEARNING = "learning"
    PSYCHOLOGICAL = "psychological"
    CAREER = "career"
    SELF_REALIZATION = "self_realization"
    FINANCES = "finances"


class ActivityOptimizer:
    """
    Оптимизатор активностей для ML моделей.
    Генерирует чистые числовые данные без текстовых описаний.
    """

    def __init__(self):
        # Веса для ML-признаков на основе magic profile
        self.ml_feature_weights = {
            ActivityType.PHYSICAL: {
                'willpower_determination': 0.25,
                'willpower_persistence': 0.20,
                'emotional_stability': 0.15,
                'physical_energy': 0.40
            },
            ActivityType.SPIRITUAL: {
                'creative_imagination': 0.30,
                'creative_intuition': 0.25,
                'emotional_joy': 0.20,
                'ethical_honesty': 0.25
            },
            ActivityType.LEARNING: {
                'intellectual_curiosity': 0.30,
                'intellectual_learning': 0.25,
                'thinking_critical': 0.20,
                'willpower_focus': 0.25
            },
            ActivityType.PSYCHOLOGICAL: {
                'emotional_stability': 0.25,
                'emotional_self_awareness': 0.20,
                'psychological_authenticity': 0.30,
                'social_empathy': 0.25
            },
            ActivityType.CAREER: {
                'willpower_determination': 0.20,
                'intellectual_learning': 0.20,
                'ethical_responsibility': 0.25,
                'social_diplomacy': 0.35
            },
            ActivityType.SELF_REALIZATION: {
                'creative_imagination': 0.35,
                'creative_innovation': 0.25,
                'psychological_authenticity': 0.20,
                'intellectual_curiosity': 0.20
            },
            ActivityType.FINANCES: {
                'ethical_honesty': 0.25,
                'ethical_responsibility': 0.25,
                'intellectual_learning': 0.20,
                'willpower_persistence': 0.30
            }
        }

        # Балансировочные коэффициенты для равномерного распределения
        self.balancing_factors = {
            ActivityType.PHYSICAL: 1.0,
            ActivityType.SPIRITUAL: 1.1,
            ActivityType.LEARNING: 1.0,
            ActivityType.PSYCHOLOGICAL: 1.2,
            ActivityType.CAREER: 0.9,
            ActivityType.SELF_REALIZATION: 1.1,
            ActivityType.FINANCES: 0.8
        }

    async def calculate_optimal_activities(self, telegram_id: int, target_date: date = None) -> Dict[str, Any]:
        """
        Расчет оптимальных активностей для ML моделей.
        Возвращает только числовые данные.

        Args:
            telegram_id: ID пользователя в Telegram
            target_date: Дата для расчета (по умолчанию сегодня)

        Returns:
            Словарь с числовыми данными для ML моделей
        """
        try:
            if target_date is None:
                target_date = date.today()

            logger.info(f"🔄 Расчет ML-активностей для {telegram_id} на {target_date}")

            # Получаем magic profile - основного источника данных
            magic_profile = await magic_profile_service.get_or_calculate_magic_profile(telegram_id)

            # Валидация критичных данных
            if not self._validate_magic_profile(magic_profile):
                raise ValueError("Невалидный magic profile для расчета активностей")

            logger.info(f"✅ Magic profile получен для расчета активностей {telegram_id}")

            # Расчет ML-признаков и оценок
            ml_features = self._extract_ml_features(magic_profile)
            activity_scores = self._calculate_activity_scores(ml_features)
            optimal_indices = self._select_optimal_activities(activity_scores)

            # Формирование чистых данных для ML
            result = {
                'telegram_id': telegram_id,
                'calculation_date': target_date.isoformat(),
                'ml_features': ml_features,
                'activity_scores': activity_scores,
                'optimal_activities': optimal_indices,
                'feature_vector': self._create_feature_vector(ml_features, activity_scores, optimal_indices),
                'metadata': {
                    'calculated_at': datetime.now().isoformat(),
                    'algorithm_version': 'ml_1.0',
                    'data_source': 'magic_profile'
                }
            }

            # Сохранение в БД для кэширования
            await self._save_ml_activities_to_db(telegram_id, target_date, result)

            logger.info(f"✅ ML-активности рассчитаны для {telegram_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Ошибка расчета ML-активностей для {telegram_id}: {e}")
            # Возвращаем структуру по умолчанию для ML модели
            return self._get_default_ml_activities(telegram_id, target_date)

    def _validate_magic_profile(self, magic_profile: Dict) -> bool:
        """Валидация magic profile для расчета ML-активностей"""
        try:
            required_sections = [
                'ethical_framework', 'social_predispositions', 'emotional_architecture',
                'intellectual_traits', 'willpower_profile', 'creative_intuitive'
            ]

            for section in required_sections:
                if section not in magic_profile or not magic_profile[section]:
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка валидации magic profile: {e}")
            return False

    def _extract_ml_features(self, magic_profile: Dict) -> Dict[str, float]:
        """
        Извлечение чистых ML-признаков из magic profile.
        Только числовые значения, нормализованные 0-1.
        """
        try:
            features = {}

            # Ethical Framework features
            ethical = magic_profile.get('ethical_framework', {})
            features['ethical_honesty'] = ethical.get('honesty_tendency', 0.5)
            features['ethical_discretion'] = ethical.get('discretion_level', 0.5)
            features['ethical_responsibility'] = ethical.get('responsibility_capacity', 0.5)
            features['ethical_loyalty'] = ethical.get('loyalty_expression', 0.5)

            # Social Predispositions features
            social = magic_profile.get('social_predispositions', {})
            features['social_extroversion'] = social.get('extroversion_level', 0.5)
            features['social_empathy'] = social.get('empathy_capacity', 0.5)
            features['social_group_skill'] = social.get('group_dynamics_skill', 0.5)

            # Emotional Architecture features
            emotional = magic_profile.get('emotional_architecture', {})
            features['emotional_stability'] = emotional.get('emotional_stability', 0.5)
            features['emotional_vulnerability'] = emotional.get('vulnerability_comfort', 0.5)
            features['emotional_joy'] = emotional.get('joy_capacity', 0.5)

            # Intellectual Traits features
            intellectual = magic_profile.get('intellectual_traits', {})
            features['intellectual_curiosity'] = intellectual.get('curiosity_level', 0.5)
            features['intellectual_skepticism'] = intellectual.get('skepticism_tendency', 0.5)
            features['intellectual_learning'] = intellectual.get('learning_agility', 0.5)
            features['intellectual_retention'] = intellectual.get('knowledge_retention', 0.5)

            # Willpower Profile features
            willpower = magic_profile.get('willpower_profile', {})
            features['willpower_determination'] = willpower.get('determination_strength', 0.5)
            features['willpower_persistence'] = willpower.get('persistence_capacity', 0.5)
            features['willpower_adaptability'] = willpower.get('adaptability_speed', 0.5)
            features['willpower_initiative'] = willpower.get('initiative_taking', 0.5)

            # Creative Intuitive features
            creative = magic_profile.get('creative_intuitive', {})
            features['creative_imagination'] = creative.get('imagination_vividness', 0.5)
            features['creative_intuition'] = creative.get('intuition_strength', 0.5)
            features['creative_innovation'] = creative.get('innovation_capacity', 0.5)
            features['creative_artistic'] = creative.get('artistic_sensitivity', 0.5)

            # Thinking Patterns features
            thinking_patterns = intellectual.get('thinking_patterns', {})
            features['thinking_critical'] = thinking_patterns.get('critical_thinking', 0.5)
            features['thinking_creative'] = thinking_patterns.get('creative_synthesis', 0.5)
            features['thinking_systemic'] = thinking_patterns.get('systemic_thinking', 0.5)
            features['thinking_practical'] = thinking_patterns.get('practical_application', 0.5)

            # Interaction Patterns features
            interaction_patterns = social.get('interaction_patterns', {})
            features['interaction_assertiveness'] = interaction_patterns.get('assertiveness', 0.5)
            features['interaction_diplomacy'] = interaction_patterns.get('diplomacy_skill', 0.5)
            features['interaction_listening'] = interaction_patterns.get('listening_ability', 0.5)
            features['interaction_boundaries'] = interaction_patterns.get('boundary_setting', 0.5)

            # Regulation Patterns features
            regulation_patterns = emotional.get('regulation_patterns', {})
            features['regulation_self_awareness'] = regulation_patterns.get('self_awareness', 0.5)
            features['regulation_impulse_control'] = regulation_patterns.get('impulse_control', 0.5)
            features['regulation_stress_resilience'] = regulation_patterns.get('stress_resilience', 0.5)
            features['regulation_mood_consistency'] = regulation_patterns.get('mood_consistency', 0.5)

            # Execution Traits features
            execution_traits = willpower.get('execution_traits', {})
            features['execution_procrastination'] = 1.0 - execution_traits.get('procrastination_tendency',
                                                                               0.5)  # Инвертируем
            features['execution_follow_through'] = execution_traits.get('follow_through_ability', 0.5)
            features['execution_multitasking'] = execution_traits.get('multitasking_capacity', 0.5)
            features['execution_focus'] = execution_traits.get('focus_depth', 0.5)

            # Inspiration Patterns features
            inspiration_patterns = creative.get('inspiration_patterns', {})
            features['inspiration_dreams'] = inspiration_patterns.get('dream_utilization', 0.5)
            features['inspiration_symbols'] = inspiration_patterns.get('symbol_interpretation', 0.5)
            features['inspiration_patterns'] = inspiration_patterns.get('pattern_recognition', 0.5)
            features['inspiration_synthesis'] = inspiration_patterns.get('cross_domain_synthesis', 0.5)

            # Psychological Blueprint features
            psychological = magic_profile.get('psychological_blueprint', {})
            core_personality = psychological.get('core_personality', {})
            features['psychological_integrity'] = core_personality.get('integrity_index', 0.5)
            features['psychological_openness'] = core_personality.get('openness_balance', 0.5)
            features['psychological_dependability'] = core_personality.get('dependability_score', 0.5)
            features['psychological_authenticity'] = core_personality.get('authenticity_level', 0.5)

            # Нормализация всех признаков
            normalized_features = {}
            for key, value in features.items():
                normalized_features[key] = round(float(value), 6)

            return normalized_features

        except Exception as e:
            logger.error(f"❌ Ошибка извлечения ML-признаков: {e}")
            return self._get_default_ml_features()

    def _calculate_activity_scores(self, ml_features: Dict[str, float]) -> Dict[str, float]:
        """
        Расчет оценок активностей на основе ML-признаков.
        Чисто математические вычисления.
        """
        try:
            activity_scores = {}

            for activity_type in ActivityType:
                weights = self.ml_feature_weights.get(activity_type, {})
                balancing_factor = self.balancing_factors.get(activity_type, 1.0)

                total_score = 0.0
                total_weight = 0.0

                for feature_name, weight in weights.items():
                    if feature_name in ml_features:
                        feature_value = ml_features[feature_name]
                        total_score += feature_value * weight
                        total_weight += weight

                if total_weight > 0:
                    base_score = total_score / total_weight
                    # Применяем балансировочный коэффициент и сигмоиду для нормализации
                    final_score = self._sigmoid(base_score * balancing_factor * 2 - 1)
                    activity_scores[activity_type.value] = round(final_score, 6)
                else:
                    activity_scores[activity_type.value] = 0.5  # Нейтральное значение

            return activity_scores

        except Exception as e:
            logger.error(f"❌ Ошибка расчета оценок активностей: {e}")
            return self._get_default_activity_scores()

    def _select_optimal_activities(self, activity_scores: Dict[str, float]) -> List[int]:
        """
        Выбор оптимальных активностей для ML модели.
        Возвращает индексы топ-3 активностей + финансы.
        """
        try:
            # Сортируем активности по оценкам (кроме финансов)
            non_finance_scores = {k: v for k, v in activity_scores.items() if k != ActivityType.FINANCES.value}
            sorted_activities = sorted(non_finance_scores.items(), key=lambda x: x[1], reverse=True)

            # Выбираем топ-3
            top_3_indices = []
            activity_names = [ActivityType.PHYSICAL.value, ActivityType.SPIRITUAL.value,
                              ActivityType.LEARNING.value, ActivityType.PSYCHOLOGICAL.value,
                              ActivityType.CAREER.value, ActivityType.SELF_REALIZATION.value]

            for activity_name, score in sorted_activities[:3]:
                index = activity_names.index(activity_name)
                top_3_indices.append(index)

            # Добавляем финансы как отдельную активность
            finance_index = activity_names.index(
                ActivityType.FINANCES.value) if ActivityType.FINANCES.value in activity_names else 6
            top_3_indices.append(finance_index)

            return top_3_indices

        except Exception as e:
            logger.error(f"❌ Ошибка выбора оптимальных активностей: {e}")
            return [0, 1, 2, 6]  # Дефолтные индексы

    def _create_feature_vector(self, ml_features: Dict[str, float],
                               activity_scores: Dict[str, float],
                               optimal_indices: List[int]) -> List[float]:
        """
        Создание единого вектора признаков для ML модели.
        Включает все признаки, оценки и оптимальные индексы.
        """
        try:
            feature_vector = []

            # 1. Добавляем все ML-признаки
            for feature_name in sorted(ml_features.keys()):
                feature_vector.append(ml_features[feature_name])

            # 2. Добавляем оценки активностей
            for activity_name in sorted(activity_scores.keys()):
                feature_vector.append(activity_scores[activity_name])

            # 3. Добавляем оптимальные индексы как one-hot encoding
            activity_count = len(ActivityType)
            for i in range(activity_count):
                feature_vector.append(1.0 if i in optimal_indices else 0.0)

            # 4. Добавляем мета-признаки
            feature_vector.append(len(ml_features))  # Количество признаков
            feature_vector.append(sum(activity_scores.values()) / len(activity_scores))  # Средняя оценка
            feature_vector.append(max(activity_scores.values()))  # Максимальная оценка
            feature_vector.append(min(activity_scores.values()))  # Минимальная оценка

            # Округляем все значения
            feature_vector = [round(float(x), 6) for x in feature_vector]

            return feature_vector

        except Exception as e:
            logger.error(f"❌ Ошибка создания вектора признаков: {e}")
            return [0.5] * 100  # Дефолтный вектор

    def _sigmoid(self, x: float) -> float:
        """Сигмоидальная функция для нормализации значений 0-1"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except:
            return 0.5

    async def _save_ml_activities_to_db(self, telegram_id: int, target_date: date, result: Dict[str, Any]) -> bool:
        """Сохранение ML-активностей в базу данных для кэширования"""
        try:
            async with async_session() as session:
                # Проверяем существующую запись
                existing_query = select(OptimalActivities).where(
                    OptimalActivities.telegram_id == telegram_id,
                    OptimalActivities.calculation_date == target_date
                )
                existing_result = await session.execute(existing_query)
                existing_record = existing_result.scalar_one_or_none()

                if existing_record:
                    # Обновляем существующую запись
                    existing_record.activities = result['optimal_activities']
                    existing_record.energy_scores = result['activity_scores']
                    existing_record.ml_data = {
                        'ml_features': result['ml_features'],
                        'feature_vector': result['feature_vector'],
                        'metadata': result['metadata']
                    }
                    existing_record.updated_at = func.now()
                    logger.info(f"📝 Обновлены ML-активности для {telegram_id}")
                else:
                    # Создаем новую запись
                    new_record = OptimalActivities(
                        telegram_id=telegram_id,
                        calculation_date=target_date,
                        activities=result['optimal_activities'],
                        energy_scores=result['activity_scores'],
                        ml_data={
                            'ml_features': result['ml_features'],
                            'feature_vector': result['feature_vector'],
                            'metadata': result['metadata']
                        }
                    )
                    session.add(new_record)
                    logger.info(f"🆕 Созданы ML-активности для {telegram_id}")

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения ML-активностей в БД для {telegram_id}: {e}")
            await session.rollback()
            return False

    def _get_default_ml_features(self) -> Dict[str, float]:
        """Возвращает дефолтные ML-признаки"""
        return {f'feature_{i}': 0.5 for i in range(50)}

    def _get_default_activity_scores(self) -> Dict[str, float]:
        """Возвращает дефолтные оценки активностей"""
        return {activity.value: 0.5 for activity in ActivityType}

    def _get_default_ml_activities(self, telegram_id: int, target_date: date) -> Dict[str, Any]:
        """Возвращает дефолтную структуру ML-активностей"""
        default_features = self._get_default_ml_features()
        default_scores = self._get_default_activity_scores()

        return {
            'telegram_id': telegram_id,
            'calculation_date': target_date.isoformat() if target_date else date.today().isoformat(),
            'ml_features': default_features,
            'activity_scores': default_scores,
            'optimal_activities': [0, 1, 2, 6],
            'feature_vector': [0.5] * 100,
            'metadata': {
                'calculated_at': datetime.now().isoformat(),
                'algorithm_version': 'ml_1.0_default',
                'data_source': 'default'
            }
        }


class ActivityOptimizerService:
    """Сервис для работы с оптимизатором активностей"""

    def __init__(self):
        self.optimizer = ActivityOptimizer()

    async def get_ml_activities(self, telegram_id: int, target_date: date = None) -> Dict[str, Any]:
        """
        Основной метод получения ML-активностей.
        Использует кэширование где возможно.

        Args:
            telegram_id: ID пользователя в Telegram
            target_date: Дата для расчета

        Returns:
            ML-данные для модели
        """
        try:
            if target_date is None:
                target_date = date.today()

            # Пытаемся получить кэшированные данные
            cached_data = await self._get_cached_ml_activities(telegram_id, target_date)
            if cached_data:
                logger.info(f"✅ Использованы кэшированные ML-активности для {telegram_id}")
                return cached_data

            # Если нет в кэше, рассчитываем новые
            logger.info(f"🔄 Расчет новых ML-активностей для {telegram_id}")
            new_data = await self.optimizer.calculate_optimal_activities(telegram_id, target_date)
            return new_data

        except Exception as e:
            logger.error(f"❌ Ошибка получения ML-активностей для {telegram_id}: {e}")
            return self.optimizer._get_default_ml_activities(telegram_id, target_date)

    async def _get_cached_ml_activities(self, telegram_id: int, target_date: date) -> Optional[Dict[str, Any]]:
        """Получение кэшированных ML-активностей из БД"""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(OptimalActivities).where(
                        OptimalActivities.telegram_id == telegram_id,
                        OptimalActivities.calculation_date == target_date
                    )
                )
                record = result.scalar_one_or_none()

                if record and record.ml_data:
                    # Проверяем актуальность (не старше 7 дней)
                    update_threshold = datetime.now().timestamp() - 7 * 24 * 60 * 60
                    record_updated = record.updated_at.timestamp()

                    if record_updated > update_threshold:
                        ml_data = record.ml_data
                        return {
                            'telegram_id': telegram_id,
                            'calculation_date': target_date.isoformat(),
                            'ml_features': ml_data.get('ml_features', {}),
                            'activity_scores': record.energy_scores,
                            'optimal_activities': record.activities,
                            'feature_vector': ml_data.get('feature_vector', []),
                            'metadata': ml_data.get('metadata', {})
                        }
                    else:
                        logger.info(f"🔄 Кэш ML-активностей устарел для {telegram_id}")
                        return None
                return None

        except Exception as e:
            logger.error(f"❌ Ошибка получения кэшированных ML-активностей для {telegram_id}: {e}")
            return None

    async def get_feature_vector(self, telegram_id: int, target_date: date = None) -> List[float]:
        """
        Получение только вектора признаков для ML модели.
        Оптимизированный метод для быстрого доступа.

        Args:
            telegram_id: ID пользователя в Telegram
            target_date: Дата для расчета

        Returns:
            Вектор признаков для ML модели
        """
        try:
            ml_activities = await self.get_ml_activities(telegram_id, target_date)
            return ml_activities.get('feature_vector', [])

        except Exception as e:
            logger.error(f"❌ Ошибка получения вектора признаков для {telegram_id}: {e}")
            return [0.5] * 100

    async def force_recalculate(self, telegram_id: int, target_date: date = None) -> Dict[str, Any]:
        """
        Принудительный перерасчет ML-активностей.

        Args:
            telegram_id: ID пользователя в Telegram
            target_date: Дата для расчета

        Returns:
            Пересчитанные ML-данные
        """
        try:
            logger.info(f"🔄 Принудительный перерасчет ML-активностей для {telegram_id}")

            if target_date is None:
                target_date = date.today()

            # Удаляем кэш если существует
            await self._delete_cached_activities(telegram_id, target_date)

            # Выполняем перерасчет
            new_data = await self.optimizer.calculate_optimal_activities(telegram_id, target_date)

            return new_data

        except Exception as e:
            logger.error(f"❌ Ошибка принудительного перерасчета для {telegram_id}: {e}")
            raise

    async def _delete_cached_activities(self, telegram_id: int, target_date: date) -> bool:
        """Удаление кэшированных активностей из БД"""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(OptimalActivities).where(
                        OptimalActivities.telegram_id == telegram_id,
                        OptimalActivities.calculation_date == target_date
                    )
                )
                record = result.scalar_one_or_none()

                if record:
                    await session.delete(record)
                    await session.commit()
                    logger.info(f"🗑️ Удалены кэшированные активности для {telegram_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка удаления кэшированных активностей для {telegram_id}: {e}")
            await session.rollback()
            return False

    async def validate_ml_data(self, telegram_id: int, target_date: date = None) -> Dict[str, Any]:
        """
        Валидация ML-данных для отладки и мониторинга.

        Args:
            telegram_id: ID пользователя в Telegram
            target_date: Дата для валидации

        Returns:
            Результат валидации
        """
        try:
            ml_data = await self.get_ml_activities(telegram_id, target_date)

            validation_result = {
                'is_valid': True,
                'issues': [],
                'data_quality': 'high',
                'vector_length': len(ml_data.get('feature_vector', [])),
                'features_count': len(ml_data.get('ml_features', {})),
                'activities_count': len(ml_data.get('optimal_activities', []))
            }

            # Проверка длины вектора
            feature_vector = ml_data.get('feature_vector', [])
            if len(feature_vector) < 50:
                validation_result['issues'].append(f"Короткий вектор признаков: {len(feature_vector)}")
                validation_result['data_quality'] = 'medium'

            # Проверка значений в векторе
            invalid_values = [x for x in feature_vector if not (0 <= x <= 1)]
            if invalid_values:
                validation_result['issues'].append(f"Невалидные значения в векторе: {len(invalid_values)}")
                validation_result['data_quality'] = 'low'
                validation_result['is_valid'] = False

            # Проверка оптимальных активностей
            optimal_activities = ml_data.get('optimal_activities', [])
            if len(optimal_activities) != 4:
                validation_result['issues'].append(
                    f"Неверное количество оптимальных активностей: {len(optimal_activities)}")
                validation_result['is_valid'] = False

            return validation_result

        except Exception as e:
            logger.error(f"❌ Ошибка валидации ML-данных для {telegram_id}: {e}")
            return {
                'is_valid': False,
                'issues': [f'Ошибка валидации: {str(e)}'],
                'data_quality': 'low',
                'vector_length': 0,
                'features_count': 0,
                'activities_count': 0
            }


# Глобальный экземпляр сервиса для использования в других модулях
activity_optimizer_service = ActivityOptimizerService()
backend.assistant.py:
from backend.user_services import create_or_update_user, get_user_profile, update_user_profession
from backend.chart_services import create_and_save_natal_chart, get_user_natal_chart
from backend.matrix_services import calculate_and_save_psyho_matrix, get_user_matrix
from backend.prediction_services import generate_and_save_prediction, get_todays_prediction, \
    format_prediction_for_display
from backend.biorhythm_services import calculate_and_save_biorhythms, get_user_biorhythms
from backend.magic_profile import magic_profile_service
from backend.database import async_session
from datetime import datetime, date, timedelta
from backend.moon import calculate_lunar_phase
import logging

logger = logging.getLogger(__name__)


class PersonalAssistant:
    """Главный класс помощника для управления всеми данными и рекомендациями"""

    def __init__(self):
        pass

    async def collect_user_data(self, telegram_id: int, birth_date: date, birth_time: datetime.time,
                                birth_city: str, current_city: str = None, profession: str = None,
                                job_position: str = None):
        """Сбор и сохранение всех данных пользователя"""
        try:
            logger.info(f"🔄 Начало сбора данных для пользователя {telegram_id}")

            # Используем транзакцию для атомарности операций
            async with async_session() as session:
                try:
                    # 1. Сохраняем основные данные пользователя
                    user = await create_or_update_user(
                        telegram_id=telegram_id,
                        birth_date=birth_date,
                        birth_time=birth_time,
                        birth_city=birth_city,
                        current_city=current_city,
                        profession=profession,
                        job_position=job_position
                    )
                    logger.info(f"✅ Данные пользователя сохранены")

                    # 2. Создаем натальную карту
                    birth_datetime = datetime.combine(birth_date, birth_time)
                    natal_chart = await create_and_save_natal_chart(
                        telegram_id=telegram_id,
                        city=birth_city,
                        birth_datetime=birth_datetime,
                        timezone="Europe/Moscow"
                    )
                    logger.info(f"✅ Натальная карта создана")

                    # 3. Рассчитываем психоматрицу
                    matrix_data = await calculate_and_save_psyho_matrix(telegram_id)
                    logger.info(f"✅ Психоматрица рассчитана")

                    # 4. СОЗДАЕМ MAGIC PROFILE (новый пункт!)
                    magic_profile = await magic_profile_service.calculate_and_save_magic_profile(telegram_id)
                    logger.info(f"✅ Magic Profile создан")

                    # 4. Рассчитываем биоритмы на сегодня
                    biorhythms = await calculate_and_save_biorhythms(telegram_id)
                    logger.info(f"✅ Биоритмы рассчитаны")

                    await session.commit()

                    return {
                        'success': True,
                        'message': "✅ Все данные успешно собраны и сохранены!",
                        'data_collected': {
                            'user_profile': True,
                            'natal_chart': True,
                            'psyho_matrix': True,
                            'magic_profile': True,
                            'biorhythms': True
                        }
                    }

                except Exception as e:
                    await session.rollback()
                    logger.error(f"❌ Ошибка в транзакции сбора данных для {telegram_id}: {e}")
                    raise

        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"❌ Ошибка при сборе данных: {str(e)}"
            }

    async def update_professional_info(self, telegram_id: int, current_city: str, profession: str,
                                       job_position: str = None):
        """Обновление профессиональной информации"""
        try:
            await update_user_profession(telegram_id, profession, job_position)

            # Обновляем город проживания
            user_profile = await get_user_profile(telegram_id)
            if user_profile:
                await create_or_update_user(
                    telegram_id=telegram_id,
                    birth_date=user_profile['birth_date'],
                    birth_time=user_profile['birth_time'],
                    birth_city=user_profile['birth_city'],
                    current_city=current_city,
                    profession=profession,
                    job_position=job_position
                )

            logger.info(f"✅ Профессиональные данные обновлены для {telegram_id}")
            return {
                'success': True,
                'message': "✅ Профессиональная информация успешно обновлена!"
            }

        except Exception as e:
            logger.error(f"❌ Ошибка обновления профессии для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"❌ Ошибка обновления данных: {str(e)}"
            }

    async def get_todays_recommendations(self, telegram_id: int):
        """Получение рекомендаций на сегодня"""
        try:
            target_date = date.today()
            logger.info(f"📅 Формирование рекомендаций на сегодня для {telegram_id}")

            # Генерируем предсказание на сегодня
            prediction = await generate_and_save_prediction(telegram_id, target_date)

            # Форматируем для отображения - теперь это строка
            formatted_prediction = await format_prediction_for_display(prediction)

            # Добавляем лунную фазу
            lunar_phase = calculate_lunar_phase(target_date)

            # ✅ Теперь formatted_prediction - это строка, а не список
            final_recommendations = f"{formatted_prediction}\n\n🌙 Текущая лунная фаза: {lunar_phase}"

            # Вывод рекомендаций для отладки
            print(f"Recommendations for user {telegram_id} on {target_date.isoformat()}:")
            print(final_recommendations)

            return {
                'success': True,
                'date': target_date.isoformat(),
                'recommendations': final_recommendations,  # ✅ Теперь это строка
                'raw_data': prediction,
                'lunar_phase': lunar_phase
            }

        except Exception as e:
            logger.error(f"❌ Ошибка получения рекомендаций на сегодня для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"❌ Не удалось получить рекомендации на сегодня: {str(e)}"
            }

    async def get_tomorrows_recommendations(self, telegram_id: int):
        """Получение рекомендаций на завтра"""
        try:
            tomorrow = date.today() + timedelta(days=1)
            logger.info(f"📅 Формирование рекомендаций на завтра ({tomorrow}) для {telegram_id}")

            prediction = await generate_and_save_prediction(telegram_id, tomorrow)
            formatted_prediction = await format_prediction_for_display(prediction)

            lunar_phase = calculate_lunar_phase(tomorrow)
            final_recommendations = f"{formatted_prediction}\n\n🌙 Лунная фаза на завтра: {lunar_phase}"

            print(f"Recommendations for user {telegram_id} on {tomorrow.isoformat()}:")
            print(final_recommendations)

            return {
                'success': True,
                'date': tomorrow.isoformat(),
                'recommendations': final_recommendations,
                'raw_data': prediction,
                'lunar_phase': lunar_phase
            }

        except Exception as e:
            logger.error(f"❌ Ошибка получения рекомендаций на завтра для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"❌ Не удалось получить рекомендации на завтра: {str(e)}"
            }

    async def get_date_recommendations(self, telegram_id: int, target_date: date):
        """Получение рекомендаций на выбранную дату"""
        try:
            logger.info(f"📅 Формирование рекомендаций на {target_date} для {telegram_id}")

            if target_date < date.today():
                return {
                    'success': False,
                    'message': "❌ Нельзя получить рекомендации для прошедших дат"
                }

            prediction = await generate_and_save_prediction(telegram_id, target_date)
            formatted_prediction = await format_prediction_for_display(prediction)

            lunar_phase = calculate_lunar_phase(target_date)
            final_recommendations = f"{formatted_prediction}\n\n🌙 Лунная фаза на {target_date}: {lunar_phase}"

            print(f"Recommendations for user {telegram_id} on {target_date.isoformat()}:")
            print(final_recommendations)

            return {
                'success': True,
                'date': target_date.isoformat(),
                'recommendations': final_recommendations,
                'raw_data': prediction,
                'lunar_phase': lunar_phase
            }

        except Exception as e:
            logger.error(f"❌ Ошибка получения рекомендаций на {target_date} для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"❌ Не удалось получить рекомендации на выбранную дату: {str(e)}"
            }

    async def get_user_data_status(self, telegram_id: int):
        """Проверка статуса собранных данных пользователя"""
        try:
            user_profile = await get_user_profile(telegram_id)
            natal_chart = await get_user_natal_chart(telegram_id)
            psyho_matrix = await get_user_matrix(telegram_id)
            biorhythms = await get_user_biorhythms(telegram_id)

            has_basic_data = user_profile is not None
            has_natal_chart = natal_chart is not None
            has_psyho_matrix = psyho_matrix is not None
            has_biorhythms = biorhythms is not None

            return {
                'has_basic_data': has_basic_data,
                'has_natal_chart': has_natal_chart,
                'has_psyho_matrix': has_psyho_matrix,
                'has_biorhythms': has_biorhythms,
                'is_complete': has_basic_data and has_natal_chart and has_psyho_matrix and has_biorhythms,
                'user_profile': user_profile
            }

        except Exception as e:
            logger.error(f"❌ Ошибка проверки статуса данных для {telegram_id}: {e}")
            return {
                'has_basic_data': False,
                'has_natal_chart': False,
                'has_psyho_matrix': False,
                'has_biorhythms': False,
                'is_complete': False
            }

    async def get_user_statistics(self, telegram_id: int):
        """Получение статистики пользователя"""
        try:
            from backend.prediction_services import get_prediction_statistics
            from backend.biorhythm_services import get_biorhythm_statistics

            data_status = await self.get_user_data_status(telegram_id)
            prediction_stats = await get_prediction_statistics(telegram_id)
            biorhythm_stats = await get_biorhythm_statistics(telegram_id)

            return {
                'data_status': data_status,
                'prediction_stats': prediction_stats,
                'biorhythm_stats': biorhythm_stats,
                'calculated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики для {telegram_id}: {e}")
            return {
                'data_status': {},
                'prediction_stats': {},
                'biorhythm_stats': {},
                'error': str(e)
            }

    async def cleanup_user_data(self, telegram_id: int):
        """Очистка данных пользователя (для администрирования)"""
        try:
            from backend.biorhythm_services import cleanup_old_biorhythms
            from backend.prediction_services import cleanup_old_predictions

            biorhythm_cleaned = await cleanup_old_biorhythms()
            prediction_cleaned = await cleanup_old_predictions()

            logger.info(f"🧹 Очищены данные для пользователя {telegram_id}")
            return {
                'success': True,
                'biorhythm_records_cleaned': biorhythm_cleaned,
                'prediction_records_cleaned': prediction_cleaned,
                'message': f"✅ Очищено {biorhythm_cleaned} записей биоритмов и {prediction_cleaned} предсказаний"
            }

        except Exception as e:
            logger.error(f"❌ Ошибка очистки данных для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"❌ Ошибка при очистке данных: {str(e)}"
            }

    async def validate_user_data(self, telegram_id: int):
        """Проверка корректности данных пользователя"""
        try:
            from backend.prediction_services import validate_prediction_data

            data_status = await self.get_user_data_status(telegram_id)
            prediction_valid = await validate_prediction_data(telegram_id)

            issues = []

            if not data_status['has_basic_data']:
                issues.append("Отсутствуют основные данные пользователя")
            if not data_status['has_natal_chart']:
                issues.append("Отсутствует натальная карта")
            if not data_status['has_psyho_matrix']:
                issues.append("Отсутствует психоматрица")
            if not data_status['has_biorhythms']:
                issues.append("Отсутствуют данные биоритмов")
            if not prediction_valid:
                issues.append("Некорректные данные предсказаний")

            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'data_status': data_status,
                'prediction_valid': prediction_valid
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации данных для {telegram_id}: {e}")
            return {
                'is_valid': False,
                'issues': [f"Ошибка валидации: {str(e)}"],
                'data_status': {},
                'prediction_valid': False
            }

    async def calculate_magic_profile(self, telegram_id: int):
        """Расчет и сохранение magic profile пользователя"""
        try:
            logger.info(f"🔄 Расчет magic profile для {telegram_id}")

            from backend.magic_profile import magic_profile_service
            magic_profile = await magic_profile_service.calculate_and_save_magic_profile(telegram_id)

            logger.info(f"✅ Magic profile создан для {telegram_id}")
            return {
                'success': True,
                'message': "Magic profile успешно создан",
                'profile_created': True
            }

        except Exception as e:
            logger.error(f"❌ Ошибка создания magic profile для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"Ошибка создания magic profile: {str(e)}"
            }

    async def get_optimal_activities(self, telegram_id: int, target_date: date = None):
        """Получение оптимальных активностей для пользователя"""
        try:
            if target_date is None:
                target_date = date.today()

            logger.info(f"🔄 Получение оптимальных активностей для {telegram_id}")

            from backend.activity_optimizer import activity_optimizer_service
            activities_data = await activity_optimizer_service.get_ml_activities(telegram_id, target_date)

            # Извлекаем только необходимые данные для внутреннего использования
            optimal_activities = activities_data.get('optimal_activities', [])
            activity_scores = activities_data.get('activity_scores', {})

            logger.info(f"✅ Оптимальные активности получены для {telegram_id}")
            return {
                'success': True,
                'optimal_activities': optimal_activities,
                'activity_scores': activity_scores,
                'calculation_date': target_date.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка получения оптимальных активностей для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"Ошибка получения активностей: {str(e)}"
            }

    async def validate_user_calculations(self, telegram_id: int):
        """Валидация всех расчетов пользователя"""
        try:
            logger.info(f"🔄 Валидация расчетов для {telegram_id}")

            from backend.calculation_validator import calculation_validator_service
            validation_result = await calculation_validator_service.detailed_validation_report(telegram_id)

            logger.info(f"✅ Валидация завершена для {telegram_id}")
            return {
                'success': True,
                'validation_result': validation_result
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации расчетов для {telegram_id}: {e}")
            return {
                'success': False,
                'message': f"Ошибка валидации: {str(e)}"
            }


# Создаем глобальный экземпляр помощника
assistant = PersonalAssistant()
backend.assistant_api.py:
import logging
import asyncio

from datetime import datetime, date
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from backend.database import async_session, User
from backend.activity_optimizer import activity_optimizer_service
from backend.magic_profile import magic_profile_service
from backend.calculation_validator import calculation_validator_service
from sqlalchemy.future import select

logger = logging.getLogger(__name__)


# Pydantic модели для API
class OptimalActivitiesRequest(BaseModel):
    """Запрос оптимальных активностей"""
    telegram_id: int = Field(..., description="ID пользователя в Telegram")
    date: Optional[date] = Field(None, description="Дата для расчета (по умолчанию сегодня)")


class OptimalActivitiesResponse(BaseModel):
    """Ответ с оптимальными активностями"""
    success: bool = Field(..., description="Успешность выполнения")
    telegram_id: int = Field(..., description="ID пользователя в Telegram")
    calculation_date: date = Field(..., description="Дата расчета")
    optimal_activities: List[str] = Field(..., description="Список оптимальных активностей")
    activity_scores: Dict[str, float] = Field(..., description="Оценки всех активностей")
    energy_level: float = Field(..., description="Общий уровень энергии 0-1")
    recommendations_ready: bool = Field(..., description="Готовность к рекомендациям")
    timestamp: datetime = Field(..., description="Время формирования ответа")


class UserValidationResponse(BaseModel):
    """Ответ валидации пользователя"""
    success: bool = Field(..., description="Успешность выполнения")
    telegram_id: int = Field(..., description="ID пользователя в Telegram")
    has_complete_data: bool = Field(..., description="Наличие полных данных")
    missing_data: List[str] = Field(..., description="Отсутствующие данные")
    can_calculate_activities: bool = Field(..., description="Возможность расчета активностей")
    timestamp: datetime = Field(..., description="Время проверки")


class ErrorResponse(BaseModel):
    """Ответ об ошибке"""
    success: bool = Field(False, description="Успешность выполнения")
    error: str = Field(..., description="Описание ошибки")
    error_code: str = Field(..., description="Код ошибки")
    timestamp: datetime = Field(..., description="Время ошибки")


# Создание FastAPI приложения
app = FastAPI(
    title="Personal Assistant API",
    description="API для доступа к оптимальным активностям и рекомендациям",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssistantAPIService:
    """Сервис API для проекта assistant"""

    def __init__(self):
        self.activity_mapping = {
            "physical": "физическая активность",
            "spiritual": "духовная практика",
            "learning": "обучение и развитие",
            "psychological": "психологическая работа",
            "career": "карьера и бизнес",
            "self_realization": "самореализация",
            "finances": "финансы и инвестиции"
        }

        self.activity_descriptions = {
            "physical": "Физические упражнения, спорт, здоровый образ жизни",
            "spiritual": "Медитация, йога, духовные практики, саморефлексия",
            "learning": "Изучение нового, чтение, курсы, навыки",
            "psychological": "Работа с эмоциями, терапия, самопознание",
            "career": "Рабочие задачи, проекты, карьерный рост",
            "self_realization": "Творчество, хобби, личные проекты",
            "finances": "Финансовое планирование, инвестиции, бюджет"
        }

    async def validate_user_access(self, telegram_id: int) -> bool:
        """
        Быстрая проверка доступа пользователя к API.

        Args:
            telegram_id: ID пользователя в Telegram

        Returns:
            True если пользователь существует и имеет данные
        """
        try:
            async with async_session() as session:
                # Проверяем существование пользователя
                result = await session.execute(
                    select(User).where(User.telegram_id == telegram_id)
                )
                user = result.scalar_one_or_none()

                if not user:
                    logger.warning(f"🚫 Пользователь {telegram_id} не найден")
                    return False

                # Быстрая проверка наличия основных данных
                has_birth_data = bool(user.birth_date and user.birth_time and user.birth_city)

                if not has_birth_data:
                    logger.warning(f"🚫 Пользователь {telegram_id} не имеет основных данных")
                    return False

                logger.info(f"✅ Пользователь {telegram_id} прошел валидацию")
                return True

        except Exception as e:
            logger.error(f"❌ Ошибка валидации пользователя {telegram_id}: {e}")
            return False

    async def get_optimal_activities_for_assistant(self, telegram_id: int, target_date: date = None) -> Dict[str, Any]:
        """
        Получение оптимальных активностей для assistant.
        Оптимизированная версия с минимальными данными.

        Args:
            telegram_id: ID пользователя в Telegram
            target_date: Дата для расчета

        Returns:
            Оптимизированные данные для assistant
        """
        try:
            if target_date is None:
                target_date = date.today()

            logger.info(f"🔄 Получение активностей для assistant ({telegram_id} на {target_date})")

            # Быстрая проверка пользователя
            if not await self.validate_user_access(telegram_id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Пользователь не найден или не имеет необходимых данных"
                )

            # Получаем ML-данные активностей
            ml_activities = await activity_optimizer_service.get_ml_activities(telegram_id, target_date)

            if not ml_activities:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Не удалось рассчитать активности"
                )

            # Извлекаем оптимальные активности
            optimal_indices = ml_activities.get('optimal_activities', [])
            activity_scores = ml_activities.get('activity_scores', {})

            # Преобразуем индексы в названия активностей
            activity_names = list(self.activity_mapping.keys())
            optimal_activity_names = []

            for index in optimal_indices[:3]:  # Берем только топ-3
                if index < len(activity_names):
                    optimal_activity_names.append(activity_names[index])

            # Добавляем финансы (последний элемент)
            if len(optimal_indices) > 3:
                finances_index = optimal_indices[3]
                if finances_index < len(activity_names):
                    optimal_activity_names.append(activity_names[finances_index])

            # Если что-то пошло не так, используем резервные значения
            if not optimal_activity_names:
                logger.warning(f"⚠️ Использованы резервные активности для {telegram_id}")
                optimal_activity_names = ["physical", "learning", "self_realization", "finances"]

            # Рассчитываем общий уровень энергии
            energy_level = self._calculate_energy_level(activity_scores)

            # Формируем ответ
            response = {
                "success": True,
                "telegram_id": telegram_id,
                "calculation_date": target_date.isoformat(),
                "optimal_activities": optimal_activity_names,
                "activity_scores": {k: v.get('score', 0.5) if isinstance(v, dict) else v
                                    for k, v in activity_scores.items()},
                "energy_level": round(energy_level, 4),
                "recommendations_ready": True,
                "timestamp": datetime.now()
            }

            logger.info(f"✅ Активности для assistant готовы: {optimal_activity_names}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка получения активностей для assistant ({telegram_id}): {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Внутренняя ошибка сервера: {str(e)}"
            )

    def _calculate_energy_level(self, activity_scores: Dict[str, Any]) -> float:
        """
        Расчет общего уровня энергии на основе оценок активностей.

        Args:
            activity_scores: Словарь с оценками активностей

        Returns:
            Уровень энергии 0-1
        """
        try:
            if not activity_scores:
                return 0.5

            total_score = 0.0
            count = 0

            for activity_data in activity_scores.values():
                if isinstance(activity_data, dict):
                    score = activity_data.get('score', 0.5)
                else:
                    score = activity_data

                total_score += score
                count += 1

            if count > 0:
                average_score = total_score / count
                # Нормализуем к более оптимистичной шкале
                energy_level = min(1.0, average_score * 1.2)
                return round(energy_level, 4)
            else:
                return 0.5

        except Exception as e:
            logger.error(f"❌ Ошибка расчета уровня энергии: {e}")
            return 0.5

    async def validate_user_data_completeness(self, telegram_id: int) -> Dict[str, Any]:
        """
        Проверка полноты данных пользователя для assistant.

        Args:
            telegram_id: ID пользователя в Telegram

        Returns:
            Статус полноты данных
        """
        try:
            # Проверяем базовый доступ
            has_access = await self.validate_user_access(telegram_id)
            if not has_access:
                return {
                    "success": False,
                    "telegram_id": telegram_id,
                    "has_complete_data": False,
                    "missing_data": ["basic_profile"],
                    "can_calculate_activities": False,
                    "timestamp": datetime.now()
                }

            # Проверяем наличие расчетов через быструю валидацию
            can_calculate = await calculation_validator_service.quick_validation(telegram_id)

            # Детальная проверка отсутствующих данных
            missing_data = []

            async with async_session() as session:
                user_result = await session.execute(
                    select(User).where(User.telegram_id == telegram_id)
                )
                user = user_result.scalar_one_or_none()

                if user:
                    if not user.birth_date:
                        missing_data.append("birth_date")
                    if not user.birth_time:
                        missing_data.append("birth_time")
                    if not user.birth_city:
                        missing_data.append("birth_city")

            # Проверяем magic profile
            try:
                magic_profile = await magic_profile_service.get_or_calculate_magic_profile(telegram_id)
                if not magic_profile:
                    missing_data.append("magic_profile")
            except:
                missing_data.append("magic_profile")

            # Проверяем активности
            try:
                activities = await activity_optimizer_service.get_ml_activities(telegram_id)
                if not activities:
                    missing_data.append("optimal_activities")
            except:
                missing_data.append("optimal_activities")

            has_complete_data = len(missing_data) == 0 and can_calculate

            return {
                "success": True,
                "telegram_id": telegram_id,
                "has_complete_data": has_complete_data,
                "missing_data": missing_data,
                "can_calculate_activities": can_calculate,
                "timestamp": datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка проверки данных пользователя {telegram_id}: {e}")
            return {
                "success": False,
                "telegram_id": telegram_id,
                "has_complete_data": False,
                "missing_data": ["validation_error"],
                "can_calculate_activities": False,
                "timestamp": datetime.now(),
                "error": str(e)
            }

    async def get_activity_descriptions(self, activity_names: List[str]) -> Dict[str, str]:
        """
        Получение описаний активностей для assistant.

        Args:
            activity_names: Список названий активностей

        Returns:
            Словарь с описаниями
        """
        try:
            descriptions = {}
            for activity in activity_names:
                if activity in self.activity_descriptions:
                    descriptions[activity] = self.activity_descriptions[activity]
                else:
                    descriptions[activity] = "Общая активность"

            return descriptions

        except Exception as e:
            logger.error(f"❌ Ошибка получения описаний активностей: {e}")
            return {activity: "Описание недоступно" for activity in activity_names}


# Глобальный экземпляр сервиса
assistant_api_service = AssistantAPIService()


# Dependency для проверки API ключа (можно расширить)
async def verify_api_key():
    """
    Заглушка для проверки API ключа.
    В продакшене нужно реализовать настоящую аутентификацию.
    """
    return True


# Эндпоинты API
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Personal Assistant API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now()
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    try:
        # Проверяем подключение к БД
        async with async_session() as session:
            await session.execute("SELECT 1")

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.post("/api/v1/optimal-activities", response_model=OptimalActivitiesResponse)
async def get_optimal_activities(
        request: OptimalActivitiesRequest,
        verified: bool = Depends(verify_api_key)
):
    """
    Получение оптимальных активностей для пользователя.

    Args:
        request: Запрос с ID пользователя и датой

    Returns:
        Оптимальные активности и оценки
    """
    try:
        result = await assistant_api_service.get_optimal_activities_for_assistant(
            telegram_id=request.telegram_id,
            target_date=request.date
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API Error in /optimal-activities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/api/v1/user/{telegram_id}/validate", response_model=UserValidationResponse)
async def validate_user_data(
        telegram_id: int,
        verified: bool = Depends(verify_api_key)
):
    """
    Проверка полноты данных пользователя.

    Args:
        telegram_id: ID пользователя в Telegram

    Returns:
        Статус полноты данных
    """
    try:
        result = await assistant_api_service.validate_user_data_completeness(telegram_id)
        return result

    except Exception as e:
        logger.error(f"❌ API Error in /validate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/api/v1/activities/descriptions")
async def get_activities_descriptions(
        activities: str,
        verified: bool = Depends(verify_api_key)
):
    """
    Получение описаний активностей.

    Args:
        activities: Список активностей через запятую

    Returns:
        Описания активностей
    """
    try:
        activity_list = [a.strip() for a in activities.split(",") if a.strip()]
        descriptions = await assistant_api_service.get_activity_descriptions(activity_list)

        return {
            "success": True,
            "activities": descriptions,
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"❌ API Error in /descriptions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/api/v1/user/{telegram_id}/energy-level")
async def get_user_energy_level(
        telegram_id: int,
        date: date = None,
        verified: bool = Depends(verify_api_key)
):
    """
    Получение уровня энергии пользователя.

    Args:
        telegram_id: ID пользователя в Telegram
        date: Дата для расчета

    Returns:
        Уровень энергии
    """
    try:
        activities_data = await assistant_api_service.get_optimal_activities_for_assistant(
            telegram_id=telegram_id,
            target_date=date
        )

        return {
            "success": True,
            "telegram_id": telegram_id,
            "energy_level": activities_data["energy_level"],
            "calculation_date": activities_data["calculation_date"],
            "timestamp": datetime.now()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API Error in /energy-level: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Обработчики ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик общих исключений"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now()
        ).dict()
    )


# Запуск сервера (для разработки)
if __name__ == "__main__":
    uvicorn.run(
        "backend.assistant_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


#Для использования в assistant:
#import requests

# Получение оптимальных активностей
#response = requests.post("http://astra-api:8000/api/v1/optimal-activities",
#                         json={"telegram_id": 123456789})

#activities = response.json()["optimal_activities"]  # ["physical", "learning", "career", "finances"]
#energy_level = response.json()["energy_level"]  # 0.85
backend.biorhythm_calculator.py:
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BiorhythmCalculator:
    """
    Калькулятор биоритмов на основе даты рождения.
    Рассчитывает физический, эмоциональный и интеллектуальный циклы.
    """

    def __init__(self):
        # Периоды биоритмов в днях
        self.PHYSICAL_CYCLE = 23
        self.EMOTIONAL_CYCLE = 28
        self.INTELLECTUAL_CYCLE = 33
        self.INTUITIVE_CYCLE = 38  # Дополнительный цикл

    def calculate_biorhythms(self, birth_date: date, target_date: date) -> Dict:
        """
        Расчет биоритмов на заданную дату

        Args:
            birth_date: Дата рождения
            target_date: Дата для расчета

        Returns:
            Словарь с данными биоритмов
        """
        try:
            # Вычисляем количество прожитых дней
            days_lived = (target_date - birth_date).days

            if days_lived < 0:
                raise ValueError("Дата расчета не может быть раньше даты рождения")

            # Рассчитываем фазы биоритмов
            physical = self._calculate_cycle(days_lived, self.PHYSICAL_CYCLE)
            emotional = self._calculate_cycle(days_lived, self.EMOTIONAL_CYCLE)
            intellectual = self._calculate_cycle(days_lived, self.INTELLECTUAL_CYCLE)
            intuitive = self._calculate_cycle(days_lived, self.INTUITIVE_CYCLE)

            # Общий показатель энергии
            overall_energy = self._calculate_overall_energy(physical, emotional, intellectual, intuitive)

            # Рекомендации на основе биоритмов
            recommendations = self._generate_recommendations(physical, emotional, intellectual, intuitive,
                                                             overall_energy)

            biorhythm_data = {
                'calculation_date': target_date.isoformat(),
                'days_lived': days_lived,
                'cycles': {
                    'physical': physical,
                    'emotional': emotional,
                    'intellectual': intellectual,
                    'intuitive': intuitive
                },
                'overall_energy': overall_energy,
                'recommendations': recommendations,
                'critical_days': self._find_critical_days(physical, emotional, intellectual, target_date),
                'peak_days': self._find_peak_days(physical, emotional, intellectual, target_date)
            }

            logger.info(f"✅ Биоритмы рассчитаны для {target_date}, прожито дней: {days_lived}")
            return biorhythm_data

        except Exception as e:
            logger.error(f"❌ Ошибка расчета биоритмов: {e}")
            raise

    def _calculate_cycle(self, days_lived: int, cycle_length: int) -> Dict:
        """
        Расчет одного цикла биоритма

        Args:
            days_lived: Количество прожитых дней
            cycle_length: Длина цикла в днях

        Returns:
            Данные цикла
        """
        # Текущая фаза в радианах (2π за полный цикл)
        phase = (2 * math.pi * days_lived) / cycle_length

        # Значение синусоиды (-1 до +1)
        value = math.sin(phase)

        # Процент от максимума (0% до 100%)
        percentage = ((value + 1) / 2) * 100

        # День в цикле (0 до cycle_length-1)
        day_in_cycle = days_lived % cycle_length

        return {
            'value': round(value, 4),
            'percentage': round(percentage, 2),
            'day_in_cycle': day_in_cycle,
            'phase': self._get_phase_description(value),
            'trend': self._get_trend(phase)
        }

    def _get_phase_description(self, value: float) -> str:
        """Описание фазы биоритма"""
        if value >= 0.7:
            return "пик энергии"
        elif value >= 0.3:
            return "высокая активность"
        elif value >= -0.3:
            return "нейтральная фаза"
        elif value >= -0.7:
            return "низкая активность"
        else:
            return "критическая точка"

    def _get_trend(self, phase: float) -> str:
        """Определение тренда (растет/падает)"""
        # Анализируем производную (cos(phase))
        derivative = math.cos(phase)

        if derivative > 0.1:
            return "растет"
        elif derivative < -0.1:
            return "падает"
        else:
            return "стабильно"

    def _calculate_overall_energy(self, physical: Dict, emotional: Dict, intellectual: Dict, intuitive: Dict) -> Dict:
        """Расчет общего уровня энергии"""
        # Взвешенная сумма всех циклов
        total_energy = (
                physical['value'] * 0.3 +  # Физический цикл - 30%
                emotional['value'] * 0.25 +  # Эмоциональный - 25%
                intellectual['value'] * 0.25 +  # Интеллектуальный - 25%
                intuitive['value'] * 0.2  # Интуитивный - 20%
        )

        # Нормализуем до 0-100%
        energy_percentage = ((total_energy + 1) / 2) * 100

        # Определяем уровень энергии
        if energy_percentage >= 80:
            level = "очень высокий"
            description = "Отличный день для активных действий и важных решений"
        elif energy_percentage >= 60:
            level = "высокий"
            description = "Хороший день для продуктивной работы"
        elif energy_percentage >= 40:
            level = "средний"
            description = "Стабильный день, подходит для рутинных задач"
        elif energy_percentage >= 20:
            level = "низкий"
            description = "День для отдыха и восстановления сил"
        else:
            level = "очень низкий"
            description = "Рекомендуется беречь энергию, избегать нагрузок"

        return {
            'value': round(total_energy, 4),
            'percentage': round(energy_percentage, 2),
            'level': level,
            'description': description
        }

    def _generate_recommendations(self, physical: Dict, emotional: Dict, intellectual: Dict, intuitive: Dict,
                                  overall: Dict) -> List[str]:
        """Генерация рекомендаций на основе биоритмов"""
        recommendations = []

        # Физические рекомендации
        if physical['value'] > 0.5:
            recommendations.append("💪 Идеальный день для спорта и физической активности")
        elif physical['value'] < -0.5:
            recommendations.append("🛌 Избегайте тяжелых физических нагрузок")

        # Эмоциональные рекомендации
        if emotional['value'] > 0.6:
            recommendations.append("😊 Отличное время для общения и новых знакомств")
        elif emotional['value'] < -0.4:
            recommendations.append("🧘 Контролируйте эмоции, избегайте конфликтов")

        # Интеллектуальные рекомендации
        if intellectual['value'] > 0.5:
            recommendations.append("📚 Благоприятный период для обучения и анализа")
        elif intellectual['value'] < -0.3:
            recommendations.append("📝 Отложите сложные интеллектуальные задачи")

        # Интуитивные рекомендации
        if intuitive['value'] > 0.4:
            recommendations.append("🔮 Доверяйте интуиции при принятии решений")

        # Общие рекомендации по энергии
        if overall['percentage'] > 70:
            recommendations.append("🚀 Используйте высокую энергию для важных проектов")
        elif overall['percentage'] < 30:
            recommendations.append("⚡ Экономьте силы, планируйте короткие перерывы")

        # Если рекомендаций мало, добавляем общие
        if len(recommendations) < 3:
            recommendations.extend([
                "📅 Следуйте своему естественному ритму",
                "⏰ Планируйте задачи в соответствии с энергетическими пиками",
                "💧 Пейте足够 воды для поддержания энергии"
            ])

        return recommendations[:5]  # Не более 5 рекомендаций

    def _find_critical_days(self, physical: Dict, emotional: Dict, intellectual: Dict, target_date: date) -> List[Dict]:
        """Определение критических дней (ближайшие 7 дней)"""
        critical_days = []

        # Проверяем текущий день
        if (abs(physical['value']) > 0.9 or
                abs(emotional['value']) > 0.9 or
                abs(intellectual['value']) > 0.9):
            critical_days.append({
                'date': target_date.isoformat(),
                'cycles': self._get_critical_cycles(physical, emotional, intellectual),
                'description': 'Критический день - будьте осторожны'
            })

        return critical_days

    def _find_peak_days(self, physical: Dict, emotional: Dict, intellectual: Dict, target_date: date) -> List[Dict]:
        """Определение пиковых дней (ближайшие 7 дней)"""
        peak_days = []

        # Проверяем текущий день
        if (physical['value'] > 0.8 or
                emotional['value'] > 0.8 or
                intellectual['value'] > 0.8):

            peak_cycles = []
            if physical['value'] > 0.8: peak_cycles.append('физический')
            if emotional['value'] > 0.8: peak_cycles.append('эмоциональный')
            if intellectual['value'] > 0.8: peak_cycles.append('интеллектуальный')

            peak_days.append({
                'date': target_date.isoformat(),
                'cycles': peak_cycles,
                'description': f'Пик энергии в циклах: {", ".join(peak_cycles)}'
            })

        return peak_days

    def _get_critical_cycles(self, physical: Dict, emotional: Dict, intellectual: Dict) -> List[str]:
        """Получение списка критических циклов"""
        critical = []
        if abs(physical['value']) > 0.9: critical.append('физический')
        if abs(emotional['value']) > 0.9: critical.append('эмоциональный')
        if abs(intellectual['value']) > 0.9: critical.append('интеллектуальный')
        return critical

    def calculate_weekly_forecast(self, birth_date: date, start_date: date, days: int = 7) -> List[Dict]:
        """Расчет прогноза биоритмов на несколько дней"""
        forecast = []

        for i in range(days):
            current_date = start_date + timedelta(days=i)
            biorhythms = self.calculate_biorhythms(birth_date, current_date)

            forecast.append({
                'date': current_date.isoformat(),
                'overall_energy': biorhythms['overall_energy']['percentage'],
                'physical': biorhythms['cycles']['physical']['percentage'],
                'emotional': biorhythms['cycles']['emotional']['percentage'],
                'intellectual': biorhythms['cycles']['intellectual']['percentage'],
                'is_critical': len(biorhythms['critical_days']) > 0,
                'is_peak': len(biorhythms['peak_days']) > 0
            })

        return forecast
backend.calculation_validator.py:
import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import math
from enum import Enum
from sqlalchemy.future import select
from sqlalchemy import func, and_

from backend.database import async_session, User, UserNatalChart, PsyhoMatrix, Biorhythms, UserMagicProfile, \
    OptimalActivities
from backend.chart_services import get_user_natal_chart
from backend.matrix_services import get_user_matrix
from backend.biorhythm_services import get_user_biorhythms
from backend.magic_profile import magic_profile_service
from backend.activity_optimizer import activity_optimizer_service

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Уровни строгости валидации"""
    BASIC = "basic"  # Быстрая проверка наличия данных
    STANDARD = "standard"  # Проверка структуры и диапазонов
    STRICT = "strict"  # Полная математическая проверка


class ValidationResult(Enum):
    """Результаты валидации"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"


class CalculationValidator:
    """
    Валидатор точности и корректности расчетов.
    Проверяет математическую согласованность данных.
    """

    def __init__(self):
        # Допустимые диапазоны для различных типов данных
        self.validation_ranges = {
            'birth_date': {
                'min': date(1900, 1, 1),
                'max': date.today() - timedelta(days=1)
            },
            'planet_longitude': {
                'min': 0.0,
                'max': 360.0
            },
            'biorhythm_value': {
                'min': -1.0,
                'max': 1.0
            },
            'ml_feature_value': {
                'min': 0.0,
                'max': 1.0
            },
            'matrix_digit_count': {
                'min': 0,
                'max': 8
            },
            'activity_score': {
                'min': 0.0,
                'max': 1.0
            }
        }

        # Математические константы для проверки
        self.mathematical_constants = {
            'circle_degrees': 360.0,
            'sign_degrees': 30.0,
            'total_planets': 10,
            'total_houses': 12,
            'total_zodiac_signs': 12
        }

    async def validate_user_calculations(self, telegram_id: int,
                                         validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[
        str, Any]:
        """
        Комплексная валидация всех расчетов пользователя.

        Args:
            telegram_id: ID пользователя в Telegram
            validation_level: Уровень строгости проверки

        Returns:
            Результаты валидации всех расчетов
        """
        try:
            logger.info(f"🔄 Валидация расчетов для {telegram_id} (уровень: {validation_level.value})")

            # Параллельная валидация всех типов расчетов
            validation_tasks = [
                self._validate_user_profile(telegram_id, validation_level),
                self._validate_natal_chart(telegram_id, validation_level),
                self._validate_psyho_matrix(telegram_id, validation_level),
                self._validate_biorhythms(telegram_id, validation_level),
                self._validate_magic_profile(telegram_id, validation_level),
                self._validate_optimal_activities(telegram_id, validation_level)
            ]

            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Обработка результатов
            validation_results = {}
            task_names = ['user_profile', 'natal_chart', 'psyho_matrix', 'biorhythms', 'magic_profile',
                          'optimal_activities']

            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    validation_results[name] = {
                        'status': ValidationResult.ERROR.value,
                        'issues': [f'Ошибка валидации: {str(result)}'],
                        'details': {}
                    }
                    logger.error(f"❌ Ошибка валидации {name} для {telegram_id}: {result}")
                else:
                    validation_results[name] = result

            # Общая оценка валидации
            overall_status = self._calculate_overall_validation_status(validation_results)

            final_result = {
                'telegram_id': telegram_id,
                'validation_timestamp': datetime.now().isoformat(),
                'validation_level': validation_level.value,
                'overall_status': overall_status,
                'results': validation_results,
                'recommendations': self._generate_validation_recommendations(validation_results)
            }

            logger.info(f"✅ Валидация завершена для {telegram_id}: {overall_status}")
            return final_result

        except Exception as e:
            logger.error(f"❌ Критическая ошибка валидации для {telegram_id}: {e}")
            return self._get_error_validation_result(telegram_id, str(e))

    async def _validate_user_profile(self, telegram_id: int, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Валидация профиля пользователя"""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(User).where(User.telegram_id == telegram_id)
                )
                user = result.scalar_one_or_none()

                if not user:
                    return {
                        'status': ValidationResult.ERROR.value,
                        'issues': ['Профиль пользователя не найден'],
                        'details': {}
                    }

                issues = []
                details = {
                    'birth_date': user.birth_date.isoformat() if user.birth_date else None,
                    'birth_city': user.birth_city,
                    'has_profession': bool(user.profession)
                }

                # BASIC validation - наличие критичных данных
                if validation_level.value >= ValidationLevel.BASIC.value:
                    if not user.birth_date:
                        issues.append('Отсутствует дата рождения')
                    if not user.birth_time:
                        issues.append('Отсутствует время рождения')
                    if not user.birth_city:
                        issues.append('Отсутствует город рождения')

                # STANDARD validation - проверка диапазонов
                if validation_level.value >= ValidationLevel.STANDARD.value:
                    if user.birth_date:
                        if user.birth_date < self.validation_ranges['birth_date']['min']:
                            issues.append(f'Дата рождения слишком ранняя: {user.birth_date}')
                        if user.birth_date > self.validation_ranges['birth_date']['max']:
                            issues.append(f'Дата рождения в будущем: {user.birth_date}')

                # STRICT validation - дополнительные проверки
                if validation_level.value >= ValidationLevel.STRICT.value:
                    if user.birth_city and len(user.birth_city.strip()) < 2:
                        issues.append('Некорректное название города рождения')

                status = ValidationResult.VALID.value if not issues else ValidationResult.WARNING.value

                return {
                    'status': status,
                    'issues': issues,
                    'details': details
                }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации профиля для {telegram_id}: {e}")
            return {
                'status': ValidationResult.ERROR.value,
                'issues': [f'Ошибка валидации профиля: {str(e)}'],
                'details': {}
            }

    async def _validate_natal_chart(self, telegram_id: int, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Валидация натальной карты"""
        try:
            natal_chart = await get_user_natal_chart(telegram_id)

            if not natal_chart:
                return {
                    'status': ValidationResult.ERROR.value,
                    'issues': ['Натальная карта не найдена'],
                    'details': {}
                }

            issues = []
            details = {
                'planets_count': 0,
                'houses_count': 0,
                'aspects_count': 0,
                'calculation_jd': natal_chart.get('metadata', {}).get('datetime', {}).get('jd', 0)
            }

            # BASIC validation - наличие основных разделов
            if validation_level.value >= ValidationLevel.BASIC.value:
                required_sections = ['planets', 'houses', 'angles']
                for section in required_sections:
                    if section not in natal_chart or not natal_chart[section]:
                        issues.append(f'Отсутствует раздел {section}')

            planets = natal_chart.get('planets', {})
            details['planets_count'] = len(planets)

            houses = natal_chart.get('houses', {})
            details['houses_count'] = len(houses)

            aspects = natal_chart.get('aspects', [])
            details['aspects_count'] = len(aspects)

            # STANDARD validation - проверка структурной целостности
            if validation_level.value >= ValidationLevel.STANDARD.value:
                # Проверка планет
                expected_planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter',
                                    'Saturn', 'Uranus', 'Neptune', 'Pluto']
                missing_planets = [p for p in expected_planets if p not in planets]
                if missing_planets:
                    issues.append(f'Отсутствуют планеты: {", ".join(missing_planets)}')

                # Проверка домов
                if len(houses) != self.mathematical_constants['total_houses']:
                    issues.append(f'Некорректное количество домов: {len(houses)}')

                # Проверка долгот планет
                for planet_name, planet_data in planets.items():
                    longitude = planet_data.get('longitude', 0)
                    if not (self.validation_ranges['planet_longitude']['min'] <= longitude <=
                            self.validation_ranges['planet_longitude']['max']):
                        issues.append(f'Некорректная долгота планеты {planet_name}: {longitude}')

            # STRICT validation - математическая проверка
            if validation_level.value >= ValidationLevel.STRICT.value:
                # Проверка суммы долгот в знаках (должна быть близка к 360°)
                total_longitude = sum(planet.get('longitude', 0) for planet in planets.values())
                expected_total = 180.0 * (len(planets) - 1)  # Примерное ожидание
                longitude_tolerance = 90.0

                if abs(total_longitude - expected_total) > longitude_tolerance:
                    issues.append(f'Сумма долгот планет выходит за допустимые пределы: {total_longitude}')

                # Проверка аспектов на корректность углов
                for aspect in aspects:
                    actual_angle = aspect.get('actual_angle', 0)
                    exact_angle = aspect.get('exact_angle', 0)
                    orb = aspect.get('orb', 0)

                    if abs(actual_angle - exact_angle) > orb:
                        issues.append(f'Некорректный орб аспекта: {aspect.get("point1")}-{aspect.get("point2")}')

            status = self._determine_validation_status(issues, validation_level)

            return {
                'status': status,
                'issues': issues,
                'details': details
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации натальной карты для {telegram_id}: {e}")
            return {
                'status': ValidationResult.ERROR.value,
                'issues': [f'Ошибка валидации натальной карты: {str(e)}'],
                'details': {}
            }

    async def _validate_psyho_matrix(self, telegram_id: int, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Валидация психоматрицы"""
        try:
            psyho_matrix = await get_user_matrix(telegram_id)

            if not psyho_matrix:
                return {
                    'status': ValidationResult.ERROR.value,
                    'issues': ['Психоматрица не найдена'],
                    'details': {}
                }

            issues = []
            details = {
                'matrix_digits': {},
                'basic_numbers': {},
                'characteristics_count': 0
            }

            matrix_data = psyho_matrix.get('pythagoras_matrix', {})
            details['matrix_digits'] = {k: v for k, v in matrix_data.items()}

            basic_numbers = psyho_matrix.get('basic_numbers', {})
            details['basic_numbers'] = {k: v for k, v in basic_numbers.items()}

            characteristics = psyho_matrix.get('characteristics', {})
            details['characteristics_count'] = len(characteristics)

            # BASIC validation - наличие основных данных
            if validation_level.value >= ValidationLevel.BASIC.value:
                if not matrix_data:
                    issues.append('Отсутствуют данные матрицы')
                if not basic_numbers:
                    issues.append('Отсутствуют базовые числа')

            # STANDARD validation - проверка числовых диапазонов
            if validation_level.value >= ValidationLevel.STANDARD.value:
                # Проверка цифр матрицы
                for digit, count in matrix_data.items():
                    if not (self.validation_ranges['matrix_digit_count']['min'] <= count <=
                            self.validation_ranges['matrix_digit_count']['max']):
                        issues.append(f'Некорректное количество цифры {digit}: {count}')

                # Проверка базовых чисел
                for number_name, number_value in basic_numbers.items():
                    if number_value <= 0:
                        issues.append(f'Некорректное базовое число {number_name}: {number_value}')

            # STRICT validation - математическая проверка
            if validation_level.value >= ValidationLevel.STRICT.value:
                # Проверка суммы цифр матрицы (должна быть в разумных пределах)
                total_digits = sum(matrix_data.values())
                if total_digits < 8 or total_digits > 25:
                    issues.append(f'Некорректная сумма цифр матрицы: {total_digits}')

                # Проверка согласованности базовых чисел
                first_number = basic_numbers.get('first', 0)
                second_number = basic_numbers.get('second', 0)

                if first_number > 0 and second_number > 0:
                    # Второе число должно быть суммой цифр первого
                    expected_second = sum(int(d) for d in str(first_number))
                    if expected_second != second_number:
                        issues.append(
                            f'Несогласованность базовых чисел: first={first_number}, second={second_number}, expected_second={expected_second}')

            status = self._determine_validation_status(issues, validation_level)

            return {
                'status': status,
                'issues': issues,
                'details': details
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации психоматрицы для {telegram_id}: {e}")
            return {
                'status': ValidationResult.ERROR.value,
                'issues': [f'Ошибка валидации психоматрицы: {str(e)}'],
                'details': {}
            }

    async def _validate_biorhythms(self, telegram_id: int, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Валидация биоритмов"""
        try:
            biorhythms = await get_user_biorhythms(telegram_id)

            if not biorhythms:
                return {
                    'status': ValidationResult.ERROR.value,
                    'issues': ['Биоритмы не найдены'],
                    'details': {}
                }

            issues = []
            details = {
                'calculation_date': biorhythms.get('calculation_date'),
                'days_lived': biorhythms.get('days_lived', 0),
                'cycles': {},
                'overall_energy': 0.0
            }

            cycles = biorhythms.get('cycles', {})
            details['cycles'] = {name: data.get('value', 0) for name, data in cycles.items()}

            overall_energy = biorhythms.get('overall_energy', {})
            details['overall_energy'] = overall_energy.get('percentage', 0)

            # BASIC validation - наличие основных циклов
            if validation_level.value >= ValidationLevel.BASIC.value:
                required_cycles = ['physical', 'emotional', 'intellectual']
                for cycle in required_cycles:
                    if cycle not in cycles:
                        issues.append(f'Отсутствует цикл {cycle}')

            # STANDARD validation - проверка диапазонов значений
            if validation_level.value >= ValidationLevel.STANDARD.value:
                for cycle_name, cycle_data in cycles.items():
                    value = cycle_data.get('value', 0)
                    if not (self.validation_ranges['biorhythm_value']['min'] <= value <=
                            self.validation_ranges['biorhythm_value']['max']):
                        issues.append(f'Некорректное значение цикла {cycle_name}: {value}')

                    percentage = cycle_data.get('percentage', 0)
                    if not (0 <= percentage <= 100):
                        issues.append(f'Некорректный процент цикла {cycle_name}: {percentage}')

                # Проверка общего уровня энергии
                if not (0 <= details['overall_energy'] <= 100):
                    issues.append(f'Некорректный общий уровень энергии: {details["overall_energy"]}')

            # STRICT validation - математическая проверка
            if validation_level.value >= ValidationLevel.STRICT.value:
                # Проверка дней жизни (должно быть положительное число)
                days_lived = biorhythms.get('days_lived', 0)
                if days_lived <= 0:
                    issues.append(f'Некорректное количество прожитых дней: {days_lived}')

                # Проверка согласованности циклов (не должны быть все на пике/минимуме одновременно)
                extreme_cycles = 0
                for cycle_data in cycles.values():
                    value = abs(cycle_data.get('value', 0))
                    if value > 0.9:  # Близко к экстремуму
                        extreme_cycles += 1

                if extreme_cycles >= len(cycles) - 1:  # Все циклы в экстремуме
                    issues.append('Подозрительная синхронизация биоритмических циклов')

            status = self._determine_validation_status(issues, validation_level)

            return {
                'status': status,
                'issues': issues,
                'details': details
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации биоритмов для {telegram_id}: {e}")
            return {
                'status': ValidationResult.ERROR.value,
                'issues': [f'Ошибка валидации биоритмов: {str(e)}'],
                'details': {}
            }

    async def _validate_magic_profile(self, telegram_id: int, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Валидация magic profile"""
        try:
            magic_profile = await magic_profile_service.get_or_calculate_magic_profile(telegram_id)

            if not magic_profile:
                return {
                    'status': ValidationResult.ERROR.value,
                    'issues': ['Magic profile не найден'],
                    'details': {}
                }

            issues = []
            details = {
                'sections_count': 0,
                'features_sample': {},
                'calculation_metadata': magic_profile.get('calculation_metadata', {})
            }

            # BASIC validation - наличие основных разделов
            if validation_level.value >= ValidationLevel.BASIC.value:
                required_sections = [
                    'ethical_framework', 'social_predispositions', 'emotional_architecture',
                    'intellectual_traits', 'willpower_profile', 'creative_intuitive',
                    'psychological_blueprint'
                ]

                missing_sections = []
                for section in required_sections:
                    if section not in magic_profile:
                        missing_sections.append(section)

                if missing_sections:
                    issues.append(f'Отсутствуют разделы: {", ".join(missing_sections)}')

            details['sections_count'] = len([s for s in required_sections if s in magic_profile])

            # Выборка признаков для деталей
            sample_features = {}
            for section in ['ethical_framework', 'emotional_architecture']:
                if section in magic_profile:
                    section_data = magic_profile[section]
                    for key in list(section_data.keys())[:2]:  # Первые 2 признака
                        if isinstance(section_data[key], (int, float)):
                            sample_features[f'{section}.{key}'] = section_data[key]

            details['features_sample'] = sample_features

            # STANDARD validation - проверка диапазонов значений
            if validation_level.value >= ValidationLevel.STANDARD.value:
                issues.extend(self._validate_ml_features_ranges(magic_profile))

            # STRICT validation - проверка внутренней согласованности
            if validation_level.value >= ValidationLevel.STRICT.value:
                issues.extend(self._validate_magic_profile_consistency(magic_profile))

            status = self._determine_validation_status(issues, validation_level)

            return {
                'status': status,
                'issues': issues,
                'details': details
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации magic profile для {telegram_id}: {e}")
            return {
                'status': ValidationResult.ERROR.value,
                'issues': [f'Ошибка валидации magic profile: {str(e)}'],
                'details': {}
            }

    async def _validate_optimal_activities(self, telegram_id: int, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Валидация оптимальных активностей"""
        try:
            optimal_activities = await activity_optimizer_service.get_ml_activities(telegram_id)

            if not optimal_activities:
                return {
                    'status': ValidationResult.ERROR.value,
                    'issues': ['Оптимальные активности не найдены'],
                    'details': {}
                }

            issues = []
            details = {
                'optimal_activities': optimal_activities.get('optimal_activities', []),
                'activity_scores_count': len(optimal_activities.get('activity_scores', {})),
                'feature_vector_length': len(optimal_activities.get('feature_vector', [])),
                'ml_features_count': len(optimal_activities.get('ml_features', {}))
            }

            # BASIC validation - наличие основных данных
            if validation_level.value >= ValidationLevel.BASIC.value:
                if not optimal_activities.get('optimal_activities'):
                    issues.append('Отсутствуют оптимальные активности')
                if not optimal_activities.get('activity_scores'):
                    issues.append('Отсутствуют оценки активностей')
                if not optimal_activities.get('feature_vector'):
                    issues.append('Отсутствует вектор признаков')

            # STANDARD validation - проверка структурной целостности
            if validation_level.value >= ValidationLevel.STANDARD.value:
                optimal_list = optimal_activities.get('optimal_activities', [])
                if len(optimal_list) != 4:
                    issues.append(f'Некорректное количество оптимальных активностей: {len(optimal_list)}')

                # Проверка диапазонов оценок
                activity_scores = optimal_activities.get('activity_scores', {})
                for activity, score_data in activity_scores.items():
                    score = score_data.get('score', 0) if isinstance(score_data, dict) else score_data
                    if not (self.validation_ranges['activity_score']['min'] <= score <=
                            self.validation_ranges['activity_score']['max']):
                        issues.append(f'Некорректная оценка активности {activity}: {score}')

            # STRICT validation - математическая проверка
            if validation_level.value >= ValidationLevel.STRICT.value:
                # Проверка уникальности оптимальных активностей (первые 3 должны быть уникальны)
                optimal_list = optimal_activities.get('optimal_activities', [])
                if len(optimal_list) >= 3:
                    first_three = optimal_list[:3]
                    if len(set(first_three)) != len(first_three):
                        issues.append('Дублирующиеся активности в топ-3')

                # Проверка согласованности вектора признаков
                feature_vector = optimal_activities.get('feature_vector', [])
                invalid_values = [x for x in feature_vector if not (0 <= x <= 1)]
                if invalid_values:
                    issues.append(f'Невалидные значения в векторе признаков: {len(invalid_values)}')

            status = self._determine_validation_status(issues, validation_level)

            return {
                'status': status,
                'issues': issues,
                'details': details
            }

        except Exception as e:
            logger.error(f"❌ Ошибка валидации оптимальных активностей для {telegram_id}: {e}")
            return {
                'status': ValidationResult.ERROR.value,
                'issues': [f'Ошибка валидации оптимальных активностей: {str(e)}'],
                'details': {}
            }

    def _validate_ml_features_ranges(self, data: Any, path: str = "") -> List[str]:
        """Рекурсивная проверка диапазонов ML-признаков"""
        issues = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                issues.extend(self._validate_ml_features_ranges(value, new_path))
        elif isinstance(data, (int, float)):
            if not (self.validation_ranges['ml_feature_value']['min'] <= data <=
                    self.validation_ranges['ml_feature_value']['max']):
                issues.append(f'Значение вне диапазона 0-1: {path} = {data}')
        elif isinstance(data, list):
            for i, item in enumerate(data):
                issues.extend(self._validate_ml_features_ranges(item, f"{path}[{i}]"))

        return issues

    def _validate_magic_profile_consistency(self, magic_profile: Dict) -> List[str]:
        """Проверка внутренней согласованности magic profile"""
        issues = []

        try:
            # Проверка согласованности связанных признаков
            ethical = magic_profile.get('ethical_framework', {})
            psychological = magic_profile.get('psychological_blueprint', {}).get('core_personality', {})

            honesty = ethical.get('honesty_tendency', 0.5)
            integrity = psychological.get('integrity_index', 0.5)

            # Честность и целостность должны быть коррелированы
            if abs(honesty - integrity) > 0.5:
                issues.append(f'Несогласованность честности ({honesty}) и целостности ({integrity})')

            # Проверка эмоциональной стабильности и устойчивости к стрессу
            emotional = magic_profile.get('emotional_architecture', {})
            regulation = emotional.get('regulation_patterns', {})

            stability = emotional.get('emotional_stability', 0.5)
            stress_resilience = regulation.get('stress_resilience', 0.5)

            if abs(stability - stress_resilience) > 0.4:
                issues.append(
                    f'Несогласованность стабильности ({stability}) и устойчивости к стрессу ({stress_resilience})')

        except Exception as e:
            issues.append(f'Ошибка проверки согласованности: {str(e)}')

        return issues

    def _determine_validation_status(self, issues: List[str], validation_level: ValidationLevel) -> str:
        """Определение статуса валидации на основе найденных проблем"""
        if not issues:
            return ValidationResult.VALID.value

        # Критические ошибки
        critical_keywords = ['ошибка', 'error', 'не найден', 'отсутствует', 'некорректное количество']
        critical_issues = [issue for issue in issues if any(keyword in issue.lower() for keyword in critical_keywords)]

        if critical_issues:
            return ValidationResult.ERROR.value

        # Для STRICT уровня любые проблемы - WARNING
        if validation_level == ValidationLevel.STRICT:
            return ValidationResult.WARNING.value

        # Для STANDARD уровня несколько проблем - WARNING
        if len(issues) > 2:
            return ValidationResult.WARNING.value

        return ValidationResult.VALID.value

    def _calculate_overall_validation_status(self, validation_results: Dict[str, Any]) -> str:
        """Расчет общего статуса валидации"""
        status_priority = {
            ValidationResult.ERROR.value: 3,
            ValidationResult.WARNING.value: 2,
            ValidationResult.INVALID.value: 1,
            ValidationResult.VALID.value: 0
        }

        worst_status = ValidationResult.VALID.value
        worst_priority = 0

        for result in validation_results.values():
            status = result.get('status', ValidationResult.VALID.value)
            priority = status_priority.get(status, 0)

            if priority > worst_priority:
                worst_priority = priority
                worst_status = status

        return worst_status

    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по результатам валидации"""
        recommendations = []

        for calculation_type, result in validation_results.items():
            status = result.get('status')
            issues = result.get('issues', [])

            if status == ValidationResult.ERROR.value:
                recommendations.append(f"Требуется перерасчет {calculation_type}: {', '.join(issues[:2])}")
            elif status == ValidationResult.WARNING.value and issues:
                recommendations.append(f"Рекомендуется проверить {calculation_type}: {issues[0]}")

        if not recommendations:
            recommendations.append("Все расчеты прошли валидацию успешно")

        return recommendations

    def _get_error_validation_result(self, telegram_id: int, error_message: str) -> Dict[str, Any]:
        """Возврат результата при критической ошибке"""
        return {
            'telegram_id': telegram_id,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_level': 'error',
            'overall_status': ValidationResult.ERROR.value,
            'results': {
                'system': {
                    'status': ValidationResult.ERROR.value,
                    'issues': [f'Системная ошибка: {error_message}'],
                    'details': {}
                }
            },
            'recommendations': ['Требуется системное вмешательство']
        }

    async def validate_system_health(self) -> Dict[str, Any]:
        """
        Проверка здоровья системы расчетов.

        Returns:
            Статус здоровья системы
        """
        try:
            async with async_session() as session:
                # Статистика по данным
                user_count = await session.execute(select(func.count(User.telegram_id)))
                natal_chart_count = await session.execute(select(func.count(UserNatalChart.telegram_id)))
                magic_profile_count = await session.execute(select(func.count(UserMagicProfile.telegram_id)))

                # Проверка свежести данных
                recent_calculations = await session.execute(
                    select(func.count(OptimalActivities.calculation_date)).where(
                        OptimalActivities.calculation_date >= date.today() - timedelta(days=1)
                    )
                )

                health_data = {
                    'user_count': user_count.scalar(),
                    'natal_chart_count': natal_chart_count.scalar(),
                    'magic_profile_count': magic_profile_count.scalar(),
                    'recent_calculations': recent_calculations.scalar(),
                    'check_timestamp': datetime.now().isoformat()
                }

                # Оценка здоровья системы
                issues = []
                if health_data['user_count'] == 0:
                    issues.append('Нет пользователей в системе')

                if health_data['natal_chart_count'] < health_data['user_count'] * 0.8:
                    issues.append('Многие пользователи не имеют натальных карт')

                if health_data['magic_profile_count'] < health_data['user_count'] * 0.7:
                    issues.append('Многие пользователи не имеют magic profiles')

                status = 'healthy' if not issues else 'degraded'

                return {
                    'status': status,
                    'issues': issues,
                    'data': health_data
                }

        except Exception as e:
            logger.error(f"❌ Ошибка проверки здоровья системы: {e}")
            return {
                'status': 'error',
                'issues': [f'Ошибка проверки здоровья: {str(e)}'],
                'data': {}
            }


class CalculationValidatorService:
    """Сервис для работы с валидатором расчетов"""

    def __init__(self):
        self.validator = CalculationValidator()

    async def quick_validation(self, telegram_id: int) -> bool:
        """
        Быстрая проверка валидности расчетов пользователя.

        Args:
            telegram_id: ID пользователя в Telegram

        Returns:
            True если расчеты в основном валидны
        """
        try:
            result = await self.validator.validate_user_calculations(
                telegram_id, ValidationLevel.BASIC
            )

            return result.get('overall_status') in [ValidationResult.VALID.value, ValidationResult.WARNING.value]

        except Exception as e:
            logger.error(f"❌ Ошибка быстрой валидации для {telegram_id}: {e}")
            return False

    async def detailed_validation_report(self, telegram_id: int) -> Dict[str, Any]:
        """
        Детальный отчет о валидации расчетов.

        Args:
            telegram_id: ID пользователя в Telegram

        Returns:
            Детальный отчет о валидации
        """
        return await self.validator.validate_user_calculations(
            telegram_id, ValidationLevel.STRICT
        )

    async def validate_multiple_users(self, telegram_ids: List[int]) -> Dict[str, Any]:
        """
        Валидация расчетов для нескольких пользователей.

        Args:
            telegram_ids: Список ID пользователей

        Returns:
            Агрегированные результаты валидации
        """
        try:
            validation_tasks = [
                self.validator.validate_user_calculations(tg_id, ValidationLevel.STANDARD)
                for tg_id in telegram_ids
            ]

            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Агрегация результатов
            status_count = {
                ValidationResult.VALID.value: 0,
                ValidationResult.WARNING.value: 0,
                ValidationResult.ERROR.value: 0
            }

            total_users = len(telegram_ids)
            successful_validations = 0

            for result in results:
                if not isinstance(result, Exception):
                    status = result.get('overall_status', ValidationResult.ERROR.value)
                    status_count[status] = status_count.get(status, 0) + 1

                    if status in [ValidationResult.VALID.value, ValidationResult.WARNING.value]:
                        successful_validations += 1

            success_rate = (successful_validations / total_users) * 100 if total_users > 0 else 0

            return {
                'total_users': total_users,
                'successful_validations': successful_validations,
                'success_rate': round(success_rate, 2),
                'status_distribution': status_count,
                'validation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка массовой валидации: {e}")
            return {
                'total_users': len(telegram_ids),
                'successful_validations': 0,
                'success_rate': 0.0,
                'status_distribution': {},
                'error': str(e)
            }

    async def get_system_health(self) -> Dict[str, Any]:
        """Получение статуса здоровья системы"""
        return await self.validator.validate_system_health()


# Глобальный экземпляр сервиса для использования в других модулях
calculation_validator_service = CalculationValidatorService()
backend.chart_services.py:
from backend.database import async_session, UserNatalChart
from backend.natal_chart import MLNatalChartCalculator
from sqlalchemy.future import select
import logging

logger = logging.getLogger(__name__)


async def create_and_save_natal_chart(telegram_id: int, city: str, birth_datetime, timezone: str):
    """Создание и сохранение натальной карты"""
    try:
        calculator = MLNatalChartCalculator()
        natal_data = calculator.calculate_natal_chart_ml(city, birth_datetime, timezone)

        logger.info(f"Создание натальной карты для пользователя {telegram_id}")

        async with async_session() as session:
            result = await session.execute(
                select(UserNatalChart).where(UserNatalChart.telegram_id == telegram_id)
            )
            natal_chart = result.scalar_one_or_none()

            if natal_chart:
                # Обновляем существующую натальную карту
                natal_chart.natal_data = natal_data
                logger.info(f"📝 Обновлена натальная карта для {telegram_id}")
            else:
                # Создаем новую натальную карту
                natal_chart = UserNatalChart(
                    telegram_id=telegram_id,
                    natal_data=natal_data
                )
                session.add(natal_chart)
                logger.info(f"🆕 Создана новая натальная карта для {telegram_id}")

            await session.commit()
            logger.info(f"💾 Натальная карта успешно сохранена для {telegram_id}")
            return natal_chart

    except Exception as e:
        logger.error(f"❌ Ошибка при создании натальной карты для {telegram_id}: {e}")
        raise


async def get_user_natal_chart(telegram_id: int):
    """Получение натальной карты пользователя"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(UserNatalChart).where(UserNatalChart.telegram_id == telegram_id)
            )
            natal_chart = result.scalar_one_or_none()

            if natal_chart:
                return natal_chart.natal_data
            return None

    except Exception as e:
        logger.error(f"❌ Ошибка при получении натальной карты {telegram_id}: {e}")
        return None
backend.database.py:
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, BigInteger, JSON, TIMESTAMP, String, Date, Time, Text
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from sqlalchemy import Column, BigInteger, JSON, TIMESTAMP, String, Date, Time, Text, ForeignKey
from sqlalchemy.sql import func
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://pers_assist:astra123@localhost:5432/p_assistant_bd"
)

logger.info(f"Подключаемся к БД: postgresql+asyncpg://pers_assist:******@localhost:5432/p_assistant_bd")

async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_recycle=300
)

async_session = sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession
)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    telegram_id = Column(BigInteger, primary_key=True, index=True)
    birth_date = Column(Date, nullable=False)
    birth_time = Column(Time, nullable=False)
    birth_city = Column(String(100), nullable=False)
    profession = Column(String(100), nullable=True)
    job_position = Column(String(100), nullable=True)
    current_city = Column(String(100), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, birth_date={self.birth_date})>"

class UserNatalChart(Base):
    __tablename__ = 'user_natal_charts'

    telegram_id = Column(BigInteger, ForeignKey('users.telegram_id', ondelete='CASCADE'), primary_key=True, index=True)
    natal_data = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<UserNatalChart(telegram_id={self.telegram_id})>"

class PsyhoMatrix(Base):
    __tablename__ = 'psyho_matrix'

    telegram_id = Column(BigInteger, ForeignKey('users.telegram_id', ondelete='CASCADE'), primary_key=True, index=True)
    matrix_data = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<PsyhoMatrix(telegram_id={self.telegram_id})>"

class NatalPredictions(Base):
    __tablename__ = 'natal_predictions'

    telegram_id = Column(BigInteger, ForeignKey('users.telegram_id', ondelete='CASCADE'), primary_key=True, index=True)
    predictions = Column(JSON, nullable=False)
    assistant_data = Column(JSON, nullable=False, default={})
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<NatalPredictions(telegram_id={self.telegram_id})>"


class Biorhythms(Base):
    __tablename__ = 'biorhythms'

    telegram_id = Column(BigInteger, ForeignKey('users.telegram_id', ondelete='CASCADE'), primary_key=True, index=True)
    biorhythm_data = Column(JSON, nullable=False)
    calculation_date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Biorhythms(telegram_id={self.telegram_id}, date={self.calculation_date})>"


# Добавить в существующий database.py:

class UserMagicProfile(Base):
    __tablename__ = 'user_magic_profiles'

    telegram_id = Column(BigInteger, ForeignKey('users.telegram_id', ondelete='CASCADE'), primary_key=True, index=True)
    ethical_framework = Column(JSONB, nullable=False)
    social_predispositions = Column(JSONB, nullable=False)
    emotional_architecture = Column(JSONB, nullable=False)
    intellectual_traits = Column(JSONB, nullable=False)
    willpower_profile = Column(JSONB, nullable=False)
    creative_intuitive = Column(JSONB, nullable=False)
    psychological_blueprint = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<UserMagicProfile(telegram_id={self.telegram_id})>"

class OptimalActivities(Base):
    __tablename__ = 'optimal_activities'

    telegram_id = Column(BigInteger, ForeignKey('users.telegram_id', ondelete='CASCADE'), primary_key=True, index=True)
    calculation_date = Column(Date, primary_key=True, index=True)
    activities = Column(JSONB, nullable=False)  # Список оптимальных активностей
    energy_scores = Column(JSONB, nullable=False)  # Оценки активностей
    ml_data = Column(JSONB, nullable=False, default={})  # ML-данные для кэширования
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<OptimalActivities(telegram_id={self.telegram_id}, date={self.calculation_date})>"


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
backend.db_connection.py:
from backend.database import async_session
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

async def check_db_connection():
    """Проверка подключения к базе данных"""
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        logger.info("✅ Подключение к БД успешно")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
        return False
backend.magic_profile.py:
import logging
import asyncio
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import math
from sqlalchemy.future import select
from sqlalchemy import func

from backend.database import async_session, UserMagicProfile
from backend.chart_services import get_user_natal_chart
from backend.matrix_services import get_user_matrix
from backend.biorhythm_services import get_user_biorhythms
from backend.user_services import get_user_profile

logger = logging.getLogger(__name__)


class MagicProfileCalculator:
    """
    Калькулятор психологических профилей для ML моделей.
    Создает структурированные данные на основе астрологии, нумерологии и биоритмов.
    """
    
    def __init__(self):
        self.planet_weights = {
            'Sun': 1.0, 'Moon': 0.9, 'Mercury': 0.8, 'Venus': 0.85, 'Mars': 0.8,
            'Jupiter': 0.7, 'Saturn': 0.9, 'Uranus': 0.6, 'Neptune': 0.7, 'Pluto': 0.8
        }
        
        self.house_weights = {
            1: 1.0, 2: 0.8, 3: 0.7, 4: 0.9, 5: 0.8, 6: 0.7,
            7: 0.8, 8: 0.9, 9: 0.8, 10: 1.0, 11: 0.7, 12: 0.8
        }
        
        self.element_modifiers = {
            'fire': {'assertiveness': 0.8, 'energy': 0.9, 'stability': 0.6},
            'earth': {'practicality': 0.9, 'stability': 0.8, 'creativity': 0.5},
            'air': {'communication': 0.9, 'intellect': 0.8, 'emotions': 0.6},
            'water': {'intuition': 0.9, 'emotions': 0.8, 'logic': 0.5}
        }

    async def calculate_magic_profile(self, telegram_id: int, 
                                   natal_chart: Dict = None,
                                   psyho_matrix: Dict = None,
                                   biorhythms: Dict = None,
                                   user_profile: Dict = None) -> Dict[str, Any]:
        """
        Оптимизированный метод расчета magic profile.
        Принимает готовые данные, а не запрашивает их повторно.
        
        Args:
            telegram_id: ID пользователя в Telegram
            natal_chart: Готовая натальная карта (опционально)
            psyho_matrix: Готовая психоматрица (опционально) 
            biorhythms: Готовые биоритмы (опционально)
            user_profile: Готовый профиль пользователя (опционально)
            
        Returns:
            Полный психологический профиль в формате для ML моделей
        """
        try:
            logger.info(f"🔄 Расчет magic profile для пользователя {telegram_id}")
            
            # Если данные не переданы, получаем их (резервный вариант)
            if not all([natal_chart, psyho_matrix, biorhythms, user_profile]):
                logger.info(f"🔄 Получение данных для {telegram_id} (резервный вариант)")
                natal_chart, psyho_matrix, biorhythms, user_profile = await asyncio.gather(
                    get_user_natal_chart(telegram_id),
                    get_user_matrix(telegram_id),
                    get_user_biorhythms(telegram_id),
                    get_user_profile(telegram_id),
                    return_exceptions=True
                )
            
            # Валидация полученных данных
            validation_result = self._validate_input_data(
                natal_chart, psyho_matrix, biorhythms, user_profile
            )
            if not validation_result["is_valid"]:
                raise ValueError(f"Недостаточно данных для расчета: {validation_result['missing']}")
            
            logger.info(f"✅ Все исходные данные получены для {telegram_id}")
            
            # Параллельные расчеты различных аспектов профиля
            ethical_framework, social_predispositions, emotional_architecture = await asyncio.gather(
                self._calculate_ethical_framework(natal_chart, psyho_matrix),
                self._calculate_social_predispositions(natal_chart, biorhythms),
                self._calculate_emotional_architecture(natal_chart, biorhythms),
            )
            
            intellectual_traits, willpower_profile, creative_intuitive = await asyncio.gather(
                self._calculate_intellectual_traits(natal_chart, psyho_matrix),
                self._calculate_willpower_profile(natal_chart, psyho_matrix),
                self._calculate_creative_intuitive(natal_chart, psyho_matrix),
            )
            
            # Создание интегрированной структуры
            psychological_blueprint = self._create_psychological_blueprint(
                ethical_framework, social_predispositions, emotional_architecture,
                intellectual_traits, willpower_profile, creative_intuitive
            )
            
            # Формирование финального профиля
            magic_profile = {
                'telegram_id': telegram_id,
                'ethical_framework': ethical_framework,
                'social_predispositions': social_predispositions,
                'emotional_architecture': emotional_architecture,
                'intellectual_traits': intellectual_traits,
                'willpower_profile': willpower_profile,
                'creative_intuitive': creative_intuitive,
                'psychological_blueprint': psychological_blueprint,
                'calculation_metadata': {
                    'calculated_at': datetime.now().isoformat(),
                    'data_sources': ['natal_chart', 'psyho_matrix', 'biorhythms'],
                    'version': '1.0'
                }
            }
            
            logger.info(f"✅ Magic profile успешно рассчитан для {telegram_id}")
            return magic_profile
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета magic profile для {telegram_id}: {e}")
            raise

    def _validate_input_data(self, natal_chart: Dict, psyho_matrix: Dict, 
                           biorhythms: Dict, user_profile: Dict) -> Dict[str, Any]:
        """Валидация входных данных для расчета"""
        missing = []
        
        if not natal_chart or 'planets' not in natal_chart:
            missing.append('natal_chart')
        if not psyho_matrix or 'pythagoras_matrix' not in psyho_matrix:
            missing.append('psyho_matrix')
        if not biorhythms or 'cycles' not in biorhythms:
            missing.append('biorhythms')
        if not user_profile or 'birth_date' not in user_profile:
            missing.append('user_profile')
            
        return {
            'is_valid': len(missing) == 0,
            'missing': missing
        }

    async def _calculate_ethical_framework(self, natal_chart: Dict, psyho_matrix: Dict) -> Dict[str, Any]:
        """Расчет морально-этических качеств"""
        try:
            planets = natal_chart.get('planets', {})
            houses = natal_chart.get('houses', {})
            placements = natal_chart.get('placements', {})
            
            # Анализ Saturn (ответственность)
            saturn_data = planets.get('Saturn', {})
            saturn_house = placements.get('Saturn', 1)
            saturn_influence = self._calculate_planet_influence(saturn_data, saturn_house, 'Saturn')
            
            # Анализ 9-го дома (этика, философия)
            ninth_house = houses.get(9, {})
            ninth_house_influence = self._calculate_house_influence(ninth_house, 9)
            
            # Анализ земных элементов
            earth_influence = self._calculate_element_influence(natal_chart, 'earth')
            
            # Расчет метрик на основе нумерологии
            matrix_data = psyho_matrix.get('pythagoras_matrix', {})
            responsibility_score = self._calculate_matrix_responsibility(matrix_data)
            
            ethical_framework = {
                "honesty_tendency": round(self._sigmoid(saturn_influence * 0.6 + ninth_house_influence * 0.4), 4),
                "discretion_level": round(self._sigmoid(earth_influence * 0.7 + self._get_pluto_influence(planets, placements) * 0.3), 4),
                "responsibility_capacity": round(self._sigmoid(saturn_influence * 0.8 + responsibility_score * 0.2), 4),
                "loyalty_expression": round(self._sigmoid(self._get_moon_influence(planets, placements) * 0.6 + earth_influence * 0.4), 4),
                
                "calculated_metrics": {
                    "truth_priority": round(self._sigmoid(ninth_house_influence * 0.8 + saturn_influence * 0.2), 4),
                    "privacy_need": round(self._sigmoid(self._get_pluto_influence(planets, placements) * 0.7 + earth_influence * 0.3), 4),
                    "commitment_strength": round(self._sigmoid(saturn_influence * 0.9 + responsibility_score * 0.1), 4),
                    "trust_building_speed": round(self._sigmoid(self._get_moon_influence(planets, placements) * 0.6 + earth_influence * 0.4), 4)
                }
            }
            
            return ethical_framework
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета ethical framework: {e}")
            return self._get_default_ethical_framework()



    async def _calculate_social_predispositions(self, natal_chart: Dict, biorhythms: Dict) -> Dict[str, Any]:
        """Расчет социальных паттернов на основе элементов и домов"""
        try:
            planets = natal_chart.get('planets', {})
            houses = natal_chart.get('houses', {})
            ml_features = natal_chart.get('ml_features', {})
            
            # Анализ элементного баланса
            element_balance = ml_features.get('element_balance', {})
            fire_air_elements = element_balance.get('fire', 0) + element_balance.get('air', 0)
            earth_water_elements = element_balance.get('earth', 0) + element_balance.get('water', 0)
            total_elements = sum(element_balance.values()) or 1
            
            # Анализ 11-го дома (социальные группы)
            eleventh_house = houses.get(11, {})
            eleventh_house_influence = self._calculate_house_influence(eleventh_house, 11)
            
            # Анализ Jupiter (экспансия, социальность)
            jupiter_influence = self._get_jupiter_influence(planets, natal_chart.get('placements', {}))
            
            # Учет биоритмов для текущей социальной энергии
            emotional_cycle = biorhythms.get('cycles', {}).get('emotional', {})
            emotional_value = emotional_cycle.get('value', 0)
            
            social_predispositions = {
                "extroversion_level": round(self._sigmoid(fire_air_elements / total_elements * 0.7 + eleventh_house_influence * 0.3), 4),
                "empathy_capacity": round(self._sigmoid(self._get_moon_influence(planets, natal_chart.get('placements', {})) * 0.6 + emotional_value * 0.4), 4),
                "conflict_approach": self._determine_conflict_approach(planets, ml_features),
                "group_dynamics_skill": round(self._sigmoid(eleventh_house_influence * 0.6 + jupiter_influence * 0.4), 4),
                
                "interaction_patterns": {
                    "assertiveness": round(self._sigmoid(self._get_mars_influence(planets, natal_chart.get('placements', {})) * 0.8 + fire_air_elements / total_elements * 0.2), 4),
                    "diplomacy_skill": round(self._sigmoid(self._get_venus_influence(planets, natal_chart.get('placements', {})) * 0.7 + jupiter_influence * 0.3), 4),
                    "listening_ability": round(self._sigmoid(self._get_moon_influence(planets, natal_chart.get('placements', {})) * 0.8 + emotional_value * 0.2), 4),
                    "boundary_setting": round(self._sigmoid(self._get_saturn_influence(planets, natal_chart.get('placements', {})) * 0.6 + earth_water_elements / total_elements * 0.4), 4)
                }
            }
            
            return social_predispositions
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета social predispositions: {e}")
            return self._get_default_social_predispositions()

    async def _calculate_emotional_architecture(self, natal_chart: Dict, biorhythms: Dict) -> Dict[str, Any]:
        """Расчет эмоциональных характеристик на основе Луны, аспектов и биоритмов"""
        try:
            planets = natal_chart.get('planets', {})
            aspects = natal_chart.get('aspects', [])
            placements = natal_chart.get('placements', {})
            
            # Анализ Луны (эмоции)
            moon_influence = self._get_moon_influence(planets, placements)
            
            # Анализ аспектов к Луне
            moon_aspects = [a for a in aspects if 'Moon' in [a.get('point1'), a.get('point2')]]
            moon_aspect_tension = self._calculate_aspect_tension(moon_aspects)
            
            # Анализ 8-го дома (глубинные эмоции)
            eighth_house = natal_chart.get('houses', {}).get(8, {})
            eighth_house_influence = self._calculate_house_influence(eighth_house, 8)
            
            # Текущее эмоциональное состояние из биоритмов
            emotional_cycle = biorhythms.get('cycles', {}).get('emotional', {})
            emotional_value = emotional_cycle.get('value', 0)
            emotional_stability = 1.0 - abs(emotional_value)  # Чем ближе к 0, тем стабильнее
            
            emotional_architecture = {
                "emotional_stability": round(self._sigmoid(moon_influence * 0.5 + emotional_stability * 0.3 + (1.0 - moon_aspect_tension) * 0.2), 4),
                "vulnerability_comfort": round(self._sigmoid(eighth_house_influence * 0.7 + moon_influence * 0.3), 4),
                "anger_expression": self._determine_anger_expression(planets, aspects),
                "joy_capacity": round(self._sigmoid(self._get_venus_influence(planets, placements) * 0.6 + self._get_jupiter_influence(planets, placements) * 0.4), 4),
                
                "regulation_patterns": {
                    "self_awareness": round(self._sigmoid(moon_influence * 0.5 + self._get_mercury_influence(planets, placements) * 0.5), 4),
                    "impulse_control": round(self._sigmoid(self._get_saturn_influence(planets, placements) * 0.6 + (1.0 - moon_aspect_tension) * 0.4), 4),
                    "stress_resilience": round(self._sigmoid(self._get_saturn_influence(planets, placements) * 0.5 + emotional_stability * 0.5), 4),
                    "mood_consistency": round(self._sigmoid(emotional_stability * 0.7 + moon_influence * 0.3), 4)
                }
            }
            
            return emotional_architecture
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета emotional architecture: {e}")
            return self._get_default_emotional_architecture()

    async def _calculate_intellectual_traits(self, natal_chart: Dict, psyho_matrix: Dict) -> Dict[str, Any]:
        """Расчет интеллектуальных предрасположенностей на основе Меркурия, Урана и 3-го/9-го домов"""
        try:
            planets = natal_chart.get('planets', {})
            houses = natal_chart.get('houses', {})
            placements = natal_chart.get('placements', {})
            aspects = natal_chart.get('aspects', [])
            
            # Анализ Меркурия (интеллект, коммуникация)
            mercury_influence = self._get_mercury_influence(planets, placements)
            
            # Анализ Урана (инновации, нестандартное мышление)
            uranus_influence = self._get_uranus_influence(planets, placements)
            
            # Анализ 3-го и 9-го домов (обучение, высшее образование)
            third_house_influence = self._calculate_house_influence(houses.get(3, {}), 3)
            ninth_house_influence = self._calculate_house_influence(houses.get(9, {}), 9)
            
            # Анализ аспектов к Меркурию
            mercury_aspects = [a for a in aspects if 'Mercury' in [a.get('point1'), a.get('point2')]]
            mercury_aspect_complexity = len(mercury_aspects) / 10.0  # Нормализация
            
            # Данные из нумерологии
            matrix_data = psyho_matrix.get('pythagoras_matrix', {})
            logic_score = self._calculate_matrix_logic(matrix_data)
            
            intellectual_traits = {
                "curiosity_level": round(self._sigmoid(ninth_house_influence * 0.6 + mercury_influence * 0.4), 4),
                "skepticism_tendency": round(self._sigmoid(self._get_saturn_influence(planets, placements) * 0.7 + third_house_influence * 0.3), 4),
                "learning_agility": round(self._sigmoid(mercury_influence * 0.5 + uranus_influence * 0.5), 4),
                "knowledge_retention": round(self._sigmoid(self._get_moon_influence(planets, placements) * 0.6 + self._get_saturn_influence(planets, placements) * 0.4), 4),
                
                "thinking_patterns": {
                    "critical_thinking": round(self._sigmoid(self._get_saturn_influence(planets, placements) * 0.6 + logic_score * 0.4), 4),
                    "creative_synthesis": round(self._sigmoid(uranus_influence * 0.5 + mercury_aspect_complexity * 0.5), 4),
                    "systemic_thinking": round(self._sigmoid(self._get_saturn_influence(planets, placements) * 0.7 + third_house_influence * 0.3), 4),
                    "practical_application": round(self._sigmoid(self._calculate_element_influence(natal_chart, 'earth') * 0.8 + mercury_influence * 0.2), 4)
                }
            }
            
            return intellectual_traits
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета intellectual traits: {e}")
            return self._get_default_intellectual_traits()

    async def _calculate_willpower_profile(self, natal_chart: Dict, psyho_matrix: Dict) -> Dict[str, Any]:
        """Расчет волевых качеств на основе Марса, Сатурна и 1-го дома"""
        try:
            planets = natal_chart.get('planets', {})
            houses = natal_chart.get('houses', {})
            placements = natal_chart.get('placements', {})
            
            # Анализ Марса (воля, инициатива)
            mars_influence = self._get_mars_influence(planets, placements)
            
            # Анализ Сатурна (дисциплина, настойчивость)
            saturn_influence = self._get_saturn_influence(planets, placements)
            
            # Анализ 1-го дома (личность, инициатива)
            first_house_influence = self._calculate_house_influence(houses.get(1, {}), 1)
            
            # Анализ фиксированных знаков для устойчивости
            fixed_signs_influence = self._calculate_fixed_signs_influence(natal_chart)
            
            # Данные из нумерологии
            matrix_data = psyho_matrix.get('pythagoras_matrix', {})
            determination_score = self._calculate_matrix_determination(matrix_data)
            
            willpower_profile = {
                "determination_strength": round(self._sigmoid(mars_influence * 0.6 + first_house_influence * 0.4), 4),
                "persistence_capacity": round(self._sigmoid(saturn_influence * 0.7 + fixed_signs_influence * 0.3), 4),
                "adaptability_speed": round(self._sigmoid(self._get_mercury_influence(planets, placements) * 0.6 + (1.0 - fixed_signs_influence) * 0.4), 4),
                "initiative_taking": round(self._sigmoid(mars_influence * 0.8 + first_house_influence * 0.2), 4),
                
                "execution_traits": {
                    "procrastination_tendency": round(1.0 - self._sigmoid(saturn_influence * 0.7 + determination_score * 0.3), 4),
                    "follow_through_ability": round(self._sigmoid(saturn_influence * 0.8 + fixed_signs_influence * 0.2), 4),
                    "multitasking_capacity": round(self._sigmoid(self._get_mercury_influence(planets, placements) * 0.7 + (1.0 - fixed_signs_influence) * 0.3), 4),
                    "focus_depth": round(self._sigmoid(fixed_signs_influence * 0.6 + saturn_influence * 0.4), 4)
                }
            }
            
            return willpower_profile
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета willpower profile: {e}")
            return self._get_default_willpower_profile()

    async def _calculate_creative_intuitive(self, natal_chart: Dict, psyho_matrix: Dict) -> Dict[str, Any]:
        """Расчет творческих и интуитивных способностей на основе Нептуна, Луны и 12-го дома"""
        try:
            planets = natal_chart.get('planets', {})
            houses = natal_chart.get('houses', {})
            placements = natal_chart.get('placements', {})
            ml_features = natal_chart.get('ml_features', {})
            
            # Анализ Нептуна (творчество, интуиция)
            neptune_influence = self._get_neptune_influence(planets, placements)
            
            # Анализ Луны (интуиция, подсознание)
            moon_influence = self._get_moon_influence(planets, placements)
            
            # Анализ 12-го дома (подсознание, духовность)
            twelfth_house_influence = self._calculate_house_influence(houses.get(12, {}), 12)
            
            # Анализ водных элементов
            water_influence = self._calculate_element_influence(natal_chart, 'water')
            
            # Анализ Урана (инновации, оригинальность)
            uranus_influence = self._get_uranus_influence(planets, placements)
            
            # Данные из нумерологии
            matrix_data = psyho_matrix.get('pythagoras_matrix', {})
            creativity_score = self._calculate_matrix_creativity(matrix_data)
            
            creative_intuitive = {
                "imagination_vividness": round(self._sigmoid(neptune_influence * 0.7 + twelfth_house_influence * 0.3), 4),
                "intuition_strength": round(self._sigmoid(moon_influence * 0.5 + water_influence * 0.5), 4),
                "innovation_capacity": round(self._sigmoid(uranus_influence * 0.6 + creativity_score * 0.4), 4),
                "artistic_sensitivity": round(self._sigmoid(neptune_influence * 0.7 + self._get_venus_influence(planets, placements) * 0.3), 4),
                
                "inspiration_patterns": {
                    "dream_utilization": round(self._sigmoid(twelfth_house_influence * 0.8 + moon_influence * 0.2), 4),
                    "symbol_interpretation": round(self._sigmoid(neptune_influence * 0.6 + water_influence * 0.4), 4),
                    "pattern_recognition": round(self._sigmoid(self._get_mercury_influence(planets, placements) * 0.7 + uranus_influence * 0.3), 4),
                    "cross_domain_synthesis": round(self._sigmoid(self._get_jupiter_influence(planets, placements) * 0.6 + uranus_influence * 0.4), 4)
                }
            }
            
            return creative_intuitive
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета creative intuitive: {e}")
            return self._get_default_creative_intuitive()

    def _create_psychological_blueprint(self, ethical_framework: Dict, social_predispositions: Dict,
                                      emotional_architecture: Dict, intellectual_traits: Dict,
                                      willpower_profile: Dict, creative_intuitive: Dict) -> Dict[str, Any]:
        """Создание интегрированной психологической структуры для ML моделей"""
        
        # Извлечение ключевых метрик из всех компонентов
        ethical_metrics = ethical_framework.get('calculated_metrics', {})
        social_patterns = social_predispositions.get('interaction_patterns', {})
        emotional_patterns = emotional_architecture.get('regulation_patterns', {})
        thinking_patterns = intellectual_traits.get('thinking_patterns', {})
        execution_traits = willpower_profile.get('execution_traits', {})
        inspiration_patterns = creative_intuitive.get('inspiration_patterns', {})
        
        psychological_blueprint = {
            "core_personality": {
                "integrity_index": round((ethical_framework.get('honesty_tendency', 0.5) + 
                                        ethical_framework.get('responsibility_capacity', 0.5)) / 2, 4),
                "openness_balance": round((social_predispositions.get('extroversion_level', 0.5) + 
                                         (1.0 - ethical_framework.get('discretion_level', 0.5))) / 2, 4),
                "dependability_score": round(ethical_framework.get('responsibility_capacity', 0.5) * 0.7 + 
                                           willpower_profile.get('follow_through_ability', 0.5) * 0.3, 4),
                "authenticity_level": round((emotional_architecture.get('emotional_stability', 0.5) + 
                                           social_patterns.get('directness_comfort', 0.5)) / 2, 4)
            },
            
            "social_architecture": {
                "trust_dynamics": {
                    "trust_giving_speed": round(social_predispositions.get('extroversion_level', 0.5) * 0.6 + 
                                              emotional_architecture.get('vulnerability_comfort', 0.5) * 0.4, 4),
                    "trust_earning_need": round(ethical_metrics.get('trust_building_speed', 0.5) * 0.7 + 
                                              social_patterns.get('boundary_setting', 0.5) * 0.3, 4),
                    "betrayal_resilience": round((1.0 - emotional_architecture.get('vulnerability_comfort', 0.5)) * 0.6 + 
                                               willpower_profile.get('persistence_capacity', 0.5) * 0.4, 4),
                    "loyalty_expression": round(ethical_framework.get('loyalty_expression', 0.5) * 0.8 + 
                                              social_patterns.get('diplomacy_skill', 0.5) * 0.2, 4)
                },
                
                "communication_ethics": {
                    "transparency_preference": round(ethical_framework.get('honesty_tendency', 0.5) * 0.7 + 
                                                   social_patterns.get('directness_comfort', 0.5) * 0.3, 4),
                    "diplomacy_priority": round(social_patterns.get('diplomacy_skill', 0.5) * 0.6 + 
                                              (1.0 - ethical_framework.get('honesty_tendency', 0.5)) * 0.4, 4),
                    "confidentiality_respect": round(ethical_framework.get('discretion_level', 0.5) * 0.8 + 
                                                   ethical_metrics.get('privacy_need', 0.5) * 0.2, 4),
                    "directness_comfort": round(social_patterns.get('assertiveness', 0.5) * 0.5 + 
                                              social_patterns.get('directness_comfort', 0.5) * 0.5, 4)
                }
            },
            
            "moral_compass": {
                "rule_following_tendency": round(ethical_framework.get('responsibility_capacity', 0.5) * 0.6 + 
                                               willpower_profile.get('follow_through_ability', 0.5) * 0.4, 4),
                "justice_sensitivity": round(intellectual_traits.get('critical_thinking', 0.5) * 0.5 + 
                                           ethical_framework.get('honesty_tendency', 0.5) * 0.5, 4),
                "forgiveness_capacity": round(emotional_architecture.get('emotional_stability', 0.5) * 0.6 + 
                                            creative_intuitive.get('intuition_strength', 0.5) * 0.4, 4),
                "consistency_importance": round(willpower_profile.get('persistence_capacity', 0.5) * 0.7 + 
                                              ethical_framework.get('responsibility_capacity', 0.5) * 0.3, 4)
            },
            
            "defense_mechanisms": {
                "vulnerability_shielding": round((1.0 - emotional_architecture.get('vulnerability_comfort', 0.5)) * 0.8 + 
                                               social_patterns.get('boundary_setting', 0.5) * 0.2, 4),
                "emotional_armor": round(emotional_patterns.get('impulse_control', 0.5) * 0.6 + 
                                       willpower_profile.get('determination_strength', 0.5) * 0.4, 4),
                "information_guarding": round(ethical_framework.get('discretion_level', 0.5) * 0.7 + 
                                            ethical_metrics.get('privacy_need', 0.5) * 0.3, 4),
                "boundary_strength": round(social_patterns.get('boundary_setting', 0.5) * 0.8 + 
                                         emotional_patterns.get('stress_resilience', 0.5) * 0.2, 4)
            },
            
            "behavioral_manifestations": {
                "promise_keeping": round(ethical_framework.get('responsibility_capacity', 0.5) * 0.8 + 
                                       willpower_profile.get('follow_through_ability', 0.5) * 0.2, 4),
                "secret_keeping_ability": round(ethical_framework.get('discretion_level', 0.5) * 0.9 + 
                                              ethical_metrics.get('privacy_need', 0.5) * 0.1, 4),
                "accountability_taking": round(ethical_framework.get('responsibility_capacity', 0.5) * 0.7 + 
                                            willpower_profile.get('determination_strength', 0.5) * 0.3, 4),
                "authentic_expression": round(emotional_architecture.get('emotional_stability', 0.5) * 0.6 + 
                                            social_predispositions.get('extroversion_level', 0.5) * 0.4, 4)
            }
        }
        
        return psychological_blueprint

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ РАСЧЕТОВ

    def _calculate_planet_influence(self, planet_data: Dict, house: int, planet_name: str) -> float:
        """Расчет влияния планеты с учетом ее положения и аспектов"""
        if not planet_data:
            return 0.5
            
        base_weight = self.planet_weights.get(planet_name, 0.5)
        house_weight = self.house_weights.get(house, 0.5)
        
        # Учет ретроградности
        retrograde_modifier = 0.8 if planet_data.get('retrograde', False) else 1.0
        
        # Учет скорости (быстрая планета - более активное влияние)
        speed = abs(planet_data.get('speed', 0))
        speed_modifier = min(1.0, speed * 10.0)  # Нормализация скорости
        
        influence = base_weight * house_weight * retrograde_modifier * (0.7 + 0.3 * speed_modifier)
        return max(0.1, min(1.0, influence))

    def _calculate_house_influence(self, house_data: Dict, house_number: int) -> float:
        """Расчет влияния дома натальной карты"""
        if not house_data:
            return 0.5
            
        base_weight = self.house_weights.get(house_number, 0.5)
        
        # Учет знака на куспиде дома
        sign_strength = self._get_sign_strength(house_data.get('sign', ''))
        
        return base_weight * sign_strength

    def _calculate_element_influence(self, natal_chart: Dict, element: str) -> float:
        """Расчет влияния элемента (огонь, земля, воздух, вода)"""
        ml_features = natal_chart.get('ml_features', {})
        element_balance = ml_features.get('element_balance', {})
        total_planets = sum(element_balance.values()) or 1
        
        element_count = element_balance.get(element, 0)
        normalized_influence = element_count / total_planets
        
        # Модификаторы для разных элементов
        element_modifiers = self.element_modifiers.get(element, {})
        avg_modifier = sum(element_modifiers.values()) / len(element_modifiers) if element_modifiers else 1.0
        
        return normalized_influence * avg_modifier

    def _get_planet_specific_influence(self, planets: Dict, placements: Dict, planet_name: str) -> float:
        """Получение специфического влияния планеты"""
        planet_data = planets.get(planet_name)
        if not planet_data:
            return 0.5
            
        house = placements.get(planet_name, 1)
        return self._calculate_planet_influence(planet_data, house, planet_name)

    def _get_moon_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Moon')

    def _get_mercury_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Mercury')

    def _get_venus_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Venus')

    def _get_mars_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Mars')

    def _get_jupiter_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Jupiter')

    def _get_saturn_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Saturn')

    def _get_uranus_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Uranus')

    def _get_neptune_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Neptune')

    def _get_pluto_influence(self, planets: Dict, placements: Dict) -> float:
        return self._get_planet_specific_influence(planets, placements, 'Pluto')

    def _calculate_aspect_tension(self, aspects: List[Dict]) -> float:
        """Расчет общего напряжения аспектов"""
        if not aspects:
            return 0.0
            
        total_tension = 0.0
        for aspect in aspects:
            aspect_type = aspect.get('aspect', '')
            strength = aspect.get('strength', 0.5)
            
            # Напряженные аспекты увеличивают напряжение
            if aspect_type in ['square', 'opposition']:
                total_tension += strength
            # Гармоничные аспекты уменьшают напряжение
            elif aspect_type in ['trine', 'sextile']:
                total_tension -= strength * 0.5
                
        return max(0.0, min(1.0, total_tension / len(aspects)))

    def _calculate_fixed_signs_influence(self, natal_chart: Dict) -> float:
        """Расчет влияния фиксированных знаков (устойчивость)"""
        planets = natal_chart.get('planets', {})
        fixed_signs = ['Taurus', 'Leo', 'Scorpio', 'Aquarius']
        
        fixed_count = 0
        total_planets = 0
        
        for planet_data in planets.values():
            if planet_data.get('sign') in fixed_signs:
                fixed_count += 1
            total_planets += 1
            
        return fixed_count / total_planets if total_planets > 0 else 0.5

    def _get_sign_strength(self, sign: str) -> float:
        """Сила влияния знака зодиака"""
        sign_strengths = {
            'Aries': 0.8, 'Taurus': 0.7, 'Gemini': 0.6, 'Cancer': 0.7,
            'Leo': 0.9, 'Virgo': 0.6, 'Libra': 0.7, 'Scorpio': 0.8,
            'Sagittarius': 0.7, 'Capricorn': 0.8, 'Aquarius': 0.7, 'Pisces': 0.6
        }
        return sign_strengths.get(sign, 0.7)

    def _determine_conflict_approach(self, planets: Dict, ml_features: Dict) -> str:
        """Определение подхода к конфликтам на основе Mars и Mercury"""
        mars_influence = self._get_mars_influence(planets, {})
        mercury_influence = self._get_mercury_influence(planets, {})
        
        if mars_influence > 0.7 and mercury_influence < 0.4:
            return "direct"
        elif mercury_influence > 0.6 and mars_influence < 0.5:
            return "analytical"
        elif mars_influence > 0.6 and mercury_influence > 0.6:
            return "strategic"
        else:
            return "adaptive"

    def _determine_anger_expression(self, planets: Dict, aspects: List[Dict]) -> str:
        """Определение выражения гнева на основе Mars и аспектов"""
        mars_influence = self._get_mars_influence(planets, {})
        saturn_influence = self._get_saturn_influence(planets, {})
        
        # Анализ аспектов к Mars
        mars_aspects = [a for a in aspects if 'Mars' in [a.get('point1'), a.get('point2')]]
        tense_aspects = [a for a in mars_aspects if a.get('aspect') in ['square', 'opposition']]
        
        if saturn_influence > 0.7 and len(tense_aspects) == 0:
            return "controlled"
        elif mars_influence > 0.7 and len(tense_aspects) > 0:
            return "explosive"
        elif mars_influence < 0.4:
            return "suppressed"
        else:
            return "balanced"

    def _calculate_matrix_responsibility(self, matrix: Dict) -> float:
        """Расчет ответственности на основе нумерологической матрицы"""
        count_4 = matrix.get('4', 0)  # Стабильность, ответственность
        count_8 = matrix.get('8', 0)  # Карма, долг
        
        responsibility_score = (count_4 * 0.6 + count_8 * 0.4) / 2.0
        return min(1.0, responsibility_score)

    def _calculate_matrix_logic(self, matrix: Dict) -> float:
        """Расчет логики на основе нумерологической матрицы"""
        count_5 = matrix.get('5', 0)  # Логика, интуиция
        count_7 = matrix.get('7', 0)  # Анализ, знания
        
        logic_score = (count_5 * 0.5 + count_7 * 0.5) / 2.0
        return min(1.0, logic_score)

    def _calculate_matrix_determination(self, matrix: Dict) -> float:
        """Расчет решительности на основе нумерологической матрицы"""
        count_1 = matrix.get('1', 0)  # Характер, воля
        count_8 = matrix.get('8', 0)  # Целеустремленность
        
        determination_score = (count_1 * 0.7 + count_8 * 0.3) / 2.0
        return min(1.0, determination_score)

    def _calculate_matrix_creativity(self, matrix: Dict) -> float:
        """Расчет креативности на основе нумерологической матрицы"""
        count_3 = matrix.get('3', 0)  # Творчество
        count_7 = matrix.get('7', 0)  # Воображение
        
        creativity_score = (count_3 * 0.6 + count_7 * 0.4) / 2.0
        return min(1.0, creativity_score)

    def _sigmoid(self, x: float) -> float:
        """Сигмоидальная функция для нормализации значений 0-1"""
        return 1.0 / (1.0 + math.exp(-10.0 * (x - 0.5)))

    # МЕТОДЫ ДЛЯ ВОЗВРАТА ЗНАЧЕНИЙ ПО УМОЛЧАНИЮ ПРИ ОШИБКАХ

    def _get_default_ethical_framework(self) -> Dict[str, Any]:
        return {
            "honesty_tendency": 0.5,
            "discretion_level": 0.5,
            "responsibility_capacity": 0.5,
            "loyalty_expression": 0.5,
            "calculated_metrics": {
                "truth_priority": 0.5,
                "privacy_need": 0.5,
                "commitment_strength": 0.5,
                "trust_building_speed": 0.5
            }
        }

    def _get_default_social_predispositions(self) -> Dict[str, Any]:
        return {
            "extroversion_level": 0.5,
            "empathy_capacity": 0.5,
            "conflict_approach": "adaptive",
            "group_dynamics_skill": 0.5,
            "interaction_patterns": {
                "assertiveness": 0.5,
                "diplomacy_skill": 0.5,
                "listening_ability": 0.5,
                "boundary_setting": 0.5
            }
        }

    def _get_default_emotional_architecture(self) -> Dict[str, Any]:
        return {
            "emotional_stability": 0.5,
            "vulnerability_comfort": 0.5,
            "anger_expression": "balanced",
            "joy_capacity": 0.5,
            "regulation_patterns": {
                "self_awareness": 0.5,
                "impulse_control": 0.5,
                "stress_resilience": 0.5,
                "mood_consistency": 0.5
            }
        }

    def _get_default_intellectual_traits(self) -> Dict[str, Any]:
        return {
            "curiosity_level": 0.5,
            "skepticism_tendency": 0.5,
            "learning_agility": 0.5,
            "knowledge_retention": 0.5,
            "thinking_patterns": {
                "critical_thinking": 0.5,
                "creative_synthesis": 0.5,
                "systemic_thinking": 0.5,
                "practical_application": 0.5
            }
        }

    def _get_default_willpower_profile(self) -> Dict[str, Any]:
        return {
            "determination_strength": 0.5,
            "persistence_capacity": 0.5,
            "adaptability_speed": 0.5,
            "initiative_taking": 0.5,
            "execution_traits": {
                "procrastination_tendency": 0.5,
                "follow_through_ability": 0.5,
                "multitasking_capacity": 0.5,
                "focus_depth": 0.5
            }
        }

    def _get_default_creative_intuitive(self) -> Dict[str, Any]:
        return {
            "imagination_vividness": 0.5,
            "intuition_strength": 0.5,
            "innovation_capacity": 0.5,
            "artistic_sensitivity": 0.5,
            "inspiration_patterns": {
                "dream_utilization": 0.5,
                "symbol_interpretation": 0.5,
                "pattern_recognition": 0.5,
                "cross_domain_synthesis": 0.5
            }
        }



class MagicProfileService:
    """Сервис для работы с magic profile в базе данных"""
    
    def __init__(self):
        self.calculator = MagicProfileCalculator()

    async def calculate_and_save_magic_profile(self, telegram_id: int) -> Dict[str, Any]:
        """
        Прямой расчет и сохранение magic profile.
        Используется при первоначальном сборе данных пользователя.
        
        Args:
            telegram_id: ID пользователя в Telegram
            
        Returns:
            Magic profile пользователя
        """
        try:
            logger.info(f"🔄 Прямой расчет magic profile для {telegram_id}")
            
            # Параллельное получение ВСЕХ необходимых данных
            natal_chart, psyho_matrix, biorhythms, user_profile = await asyncio.gather(
                get_user_natal_chart(telegram_id),
                get_user_matrix(telegram_id),
                get_user_biorhythms(telegram_id),
                get_user_profile(telegram_id),
                return_exceptions=True
            )
            
            # Расчет профиля с передачей готовых данных
            profile_data = await self.calculator.calculate_magic_profile(
                telegram_id, natal_chart, psyho_matrix, biorhythms, user_profile
            )
            
            # Сохранение в БД
            await self._save_magic_profile_to_db(telegram_id, profile_data)
            
            logger.info(f"✅ Magic profile создан и сохранен для {telegram_id}")
            return profile_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета magic profile для {telegram_id}: {e}")
            raise

    async def get_or_calculate_magic_profile(self, telegram_id: int) -> Dict[str, Any]:
        """
        Получение существующего или расчет нового magic profile.
        Для случаев, когда профиль запрашивается отдельно.
        
        Args:
            telegram_id: ID пользователя в Telegram
            
        Returns:
            Magic profile пользователя
        """
        try:
            # Сначала пытаемся получить существующий профиль
            existing_profile = await self._get_magic_profile_from_db(telegram_id)
            if existing_profile:
                logger.info(f"✅ Использован существующий magic profile для {telegram_id}")
                return existing_profile
            
            # Если профиля нет, рассчитываем новый
            logger.info(f"🔄 Расчет нового magic profile для {telegram_id}")
            return await self.calculate_and_save_magic_profile(telegram_id)
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения/расчета magic profile для {telegram_id}: {e}")
            raise

    async def _get_magic_profile_from_db(self, telegram_id: int) -> Optional[Dict[str, Any]]:
        """Получение magic profile из базы данных"""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(UserMagicProfile).where(UserMagicProfile.telegram_id == telegram_id)
                )
                profile = result.scalar_one_or_none()
                
                if profile:
                    # Проверяем актуальность данных (не старше 30 дней)
                    update_threshold = datetime.now().timestamp() - 30 * 24 * 60 * 60
                    profile_updated = profile.updated_at.timestamp()
                    
                    if profile_updated > update_threshold:
                        return {
                            'telegram_id': profile.telegram_id,
                            'ethical_framework': profile.ethical_framework,
                            'social_predispositions': profile.social_predispositions,
                            'emotional_architecture': profile.emotional_architecture,
                            'intellectual_traits': profile.intellectual_traits,
                            'willpower_profile': profile.willpower_profile,
                            'creative_intuitive': profile.creative_intuitive,
                            'psychological_blueprint': profile.psychological_blueprint,
                            'calculation_metadata': profile.psychological_blueprint.get('calculation_metadata', {})
                        }
                    else:
                        logger.info(f"🔄 Magic profile устарел для {telegram_id}, требуется перерасчет")
                        return None
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения magic profile из БД для {telegram_id}: {e}")
            return None

    async def _save_magic_profile_to_db(self, telegram_id: int, profile_data: Dict[str, Any]) -> bool:
        """Сохранение magic profile в базу данных"""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(UserMagicProfile).where(UserMagicProfile.telegram_id == telegram_id)
                )
                existing_profile = result.scalar_one_or_none()
                
                if existing_profile:
                    # Обновляем существующий профиль
                    existing_profile.ethical_framework = profile_data['ethical_framework']
                    existing_profile.social_predispositions = profile_data['social_predispositions']
                    existing_profile.emotional_architecture = profile_data['emotional_architecture']
                    existing_profile.intellectual_traits = profile_data['intellectual_traits']
                    existing_profile.willpower_profile = profile_data['willpower_profile']
                    existing_profile.creative_intuitive = profile_data['creative_intuitive']
                    existing_profile.psychological_blueprint = profile_data['psychological_blueprint']
                    existing_profile.updated_at = func.now()
                    logger.info(f"📝 Обновлен magic profile для {telegram_id}")
                else:
                    # Создаем новый профиль
                    new_profile = UserMagicProfile(
                        telegram_id=telegram_id,
                        ethical_framework=profile_data['ethical_framework'],
                        social_predispositions=profile_data['social_predispositions'],
                        emotional_architecture=profile_data['emotional_architecture'],
                        intellectual_traits=profile_data['intellectual_traits'],
                        willpower_profile=profile_data['willpower_profile'],
                        creative_intuitive=profile_data['creative_intuitive'],
                        psychological_blueprint=profile_data['psychological_blueprint']
                    )
                    session.add(new_profile)
                    logger.info(f"🆕 Создан новый magic profile для {telegram_id}")
                
                await session.commit()
                logger.info(f"💾 Magic profile сохранен в БД для {telegram_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения magic profile в БД для {telegram_id}: {e}")
            await session.rollback()
            return False

    async def force_recalculate_magic_profile(self, telegram_id: int) -> Dict[str, Any]:
        """
        Принудительный перерасчет magic profile.
        
        Args:
            telegram_id: ID пользователя в Telegram
            
        Returns:
            Пересчитанный magic profile
        """
        try:
            logger.info(f"🔄 Принудительный перерасчет magic profile для {telegram_id}")
            
            # Удаляем старый профиль если существует
            await self._delete_magic_profile_from_db(telegram_id)
            
            # Рассчитываем новый профиль
            new_profile = await self.calculate_and_save_magic_profile(telegram_id)
            
            return new_profile
            
        except Exception as e:
            logger.error(f"❌ Ошибка принудительного перерасчета magic profile для {telegram_id}: {e}")
            raise

    async def _delete_magic_profile_from_db(self, telegram_id: int) -> bool:
        """Удаление magic profile из базы данных"""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(UserMagicProfile).where(UserMagicProfile.telegram_id == telegram_id)
                )
                profile = result.scalar_one_or_none()
                
                if profile:
                    await session.delete(profile)
                    await session.commit()
                    logger.info(f"🗑️ Удален magic profile для {telegram_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка удаления magic profile из БД для {telegram_id}: {e}")
            await session.rollback()
            return False

    async def get_profile_for_ml(self, telegram_id: int) -> Dict[str, Any]:
        """
        Получение профиля в формате для ML моделей.
        
        Args:
            telegram_id: ID пользователя в Telegram
            
        Returns:
            Профиль в формате для машинного обучения
        """
        try:
            profile = await self.get_or_calculate_magic_profile(telegram_id)
            
            # Преобразуем в плоскую структуру для ML
            ml_profile = {
                'telegram_id': telegram_id,
                'features': self._flatten_profile_for_ml(profile),
                'timestamp': datetime.now().isoformat()
            }
            
            return ml_profile
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения профиля для ML для {telegram_id}: {e}")
            raise

    def _flatten_profile_for_ml(self, profile: Dict[str, Any]) -> List[float]:
        """Преобразование профиля в плоский список признаков для ML"""
        features = []
        
        # Извлекаем все числовые значения из профиля
        def extract_values(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    extract_values(value, new_prefix)
            elif isinstance(data, (int, float)):
                features.append(float(data))
            elif isinstance(data, list):
                for item in data:
                    extract_values(item, prefix)
        
        extract_values(profile)
        return features

    async def validate_profile_data(self, telegram_id: int) -> Dict[str, Any]:
        """
        Валидация данных magic profile.
        
        Args:
            telegram_id: ID пользователя в Telegram
            
        Returns:
            Результат валидации
        """
        try:
            profile = await self.get_or_calculate_magic_profile(telegram_id)
            
            validation_result = {
                'is_valid': True,
                'issues': [],
                'completeness_score': 0.0,
                'data_quality': 'high'
            }
            
            # Проверка полноты данных
            required_sections = [
                'ethical_framework', 'social_predispositions', 'emotional_architecture',
                'intellectual_traits', 'willpower_profile', 'creative_intuitive',
                'psychological_blueprint'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in profile or not profile[section]:
                    missing_sections.append(section)
                    validation_result['is_valid'] = False
            
            if missing_sections:
                validation_result['issues'].append(f"Отсутствуют разделы: {', '.join(missing_sections)}")
            
            # Проверка качества числовых данных
            numeric_issues = self._validate_numeric_data(profile)
            if numeric_issues:
                validation_result['issues'].extend(numeric_issues)
                validation_result['data_quality'] = 'medium'
            
            # Расчет оценки полноты
            completeness_score = (len(required_sections) - len(missing_sections)) / len(required_sections)
            validation_result['completeness_score'] = round(completeness_score, 4)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации magic profile для {telegram_id}: {e}")
            return {
                'is_valid': False,
                'issues': [f'Ошибка валидации: {str(e)}'],
                'completeness_score': 0.0,
                'data_quality': 'low'
            }

    def _validate_numeric_data(self, profile: Dict[str, Any]) -> List[str]:
        """Валидация числовых данных в профиле"""
        issues = []
        
        def check_numeric_values(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    check_numeric_values(value, new_path)
            elif isinstance(data, (int, float)):
                if not (0 <= data <= 1):
                    issues.append(f"Значение вне диапазона 0-1: {path} = {data}")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    check_numeric_values(item, f"{path}[{i}]")
        
        check_numeric_values(profile)
        return issues


# Глобальный экземпляр сервиса для использования в других модулях
magic_profile_service = MagicProfileService()
backend.matrix_services.py:
from backend.database import async_session, PsyhoMatrix
from backend.psyho_matrix import PsyhoMatrixCalculator
from backend.user_services import get_user_profile
from sqlalchemy.future import select
import logging

logger = logging.getLogger(__name__)


async def calculate_and_save_psyho_matrix(telegram_id: int):
    """Расчет и сохранение психоматрицы"""
    try:
        # Получаем данные пользователя
        user_profile = await get_user_profile(telegram_id)
        if not user_profile:
            raise ValueError("Пользователь не найден")

        calculator = PsyhoMatrixCalculator()
        matrix_data = calculator.calculate_matrix(user_profile['birth_date'])

        # Сохраняем психоматрицу
        async with async_session() as session:
            result = await session.execute(
                select(PsyhoMatrix).where(PsyhoMatrix.telegram_id == telegram_id)
            )
            psyho_matrix = result.scalar_one_or_none()

            if psyho_matrix:
                # Обновляем существующую психоматрицу
                psyho_matrix.matrix_data = matrix_data
                logger.info(f"📝 Обновлена психоматрица для {telegram_id}")
            else:
                # Создаем новую психоматрицу
                psyho_matrix = PsyhoMatrix(
                    telegram_id=telegram_id,
                    matrix_data=matrix_data
                )
                session.add(psyho_matrix)
                logger.info(f"🆕 Создана новая психоматрица для {telegram_id}")

            await session.commit()
            logger.info(f"✅ Психоматрица рассчитана и сохранена для {telegram_id}")

        return matrix_data

    except Exception as e:
        logger.error(f"❌ Ошибка при расчете психоматрицы для {telegram_id}: {e}")
        raise


async def get_user_matrix(telegram_id: int):
    """Получение психоматрицы пользователя"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(PsyhoMatrix).where(PsyhoMatrix.telegram_id == telegram_id)
            )
            matrix = result.scalar_one_or_none()

            if matrix:
                return matrix.matrix_data
            return None

    except Exception as e:
        logger.error(f"❌ Ошибка при получении психоматрицы {telegram_id}: {e}")
        return None
backend.moon.py:
# backend/moon.py
from datetime import date
import math

def calculate_lunar_phase(target_date: date = None) -> str:
    """Вычисляет фазу луны для заданной даты.
    Возвращает строковое описание фазы луны."""
    if target_date is None:
        target_date = date.today()

    # Используем известный алгоритм расчёта фаз луны
    # 2001-01-01 - базовая дата нового месяца
    diff = (target_date - date(2001, 1, 1)).days
    lunations = 0.20439731 + (diff * 0.03386319269)
    lunation = lunations % 1
    index = int((lunation * 8) + 0.5) & 7
    phases = [
        "New Moon",
        "Waxing Crescent",
        "First Quarter",
        "Waxing Gibbous",
        "Full Moon",
        "Waning Gibbous",
        "Last Quarter",
        "Waning Crescent",
    ]
    return phases[index]
backend.natal_chart.py:
import os
import pytz
from datetime import datetime
import swisseph as swe
from math import floor
from typing import Dict, List, Tuple, Any
import logging
import requests
import time
from urllib.parse import quote

from backend.database import async_session, UserNatalChart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLNatalChartCalculator:
    def __init__(self):
        current_dir = os.getcwd()
        ephe_path = os.path.join(current_dir, 'ephe')
        swe.set_ephe_path(ephe_path)
        swe.set_jpl_file('de441.eph')

        # Кэш для координат городов
        self.coordinates_cache = {}

        # Основные города России для быстрого доступа
        self.major_cities = {
            "москва": (55.7558, 37.6173, 156),
            "санкт-петербург": (59.9343, 30.3351, 3),
            "новосибирск": (55.0084, 82.9357, 150),
            "екатеринбург": (56.8389, 60.6057, 237),
            "нижний новгород": (56.3269, 44.0075, 78),
            "казань": (55.8304, 49.0661, 60),
            "челябинск": (55.1644, 61.4368, 228),
            "омск": (54.9884, 73.3242, 85),
            "самара": (53.2415, 50.2212, 87),
            "ростов-на-дону": (47.2225, 39.7187, 70),
            "уфа": (54.7355, 55.9587, 158),
            "красноярск": (56.0153, 92.8932, 136),
            "пермь": (58.0105, 56.2502, 149),
            "воронеж": (51.6720, 39.1843, 104),
            "волгоград": (48.7080, 44.5133, 80),
            "краснодар": (45.0355, 38.9750, 25),
            "саратов": (51.5924, 45.9608, 50),
            "тюмень": (57.1613, 65.5250, 70),
            "тольятти": (53.5088, 49.4192, 90),
            "ижевск": (56.8527, 53.2115, 140),
            "ульяновск": (54.3282, 48.3866, 80),
            "иркутск": (52.2864, 104.2806, 440),
            "хабаровск": (48.4802, 135.0719, 72),
            "ярославль": (57.6261, 39.8845, 100),
            "владивосток": (43.1332, 131.9113, 8),
            "мга": (59.7569, 31.0609, 33)
        }

        self.ORBS = {
            'conjunction': 8, 'opposition': 8, 'square': 8, 'trine': 8, 'sextile': 6,
            'quincunx': 3, 'semi-square': 3, 'semi-sextile': 3
        }

        self.planets_ml = {
            swe.SUN: 'Sun',
            swe.MOON: 'Moon',
            swe.MERCURY: 'Mercury',
            swe.VENUS: 'Venus',
            swe.MARS: 'Mars',
            swe.JUPITER: 'Jupiter',
            swe.SATURN: 'Saturn',
            swe.URANUS: 'Uranus',
            swe.NEPTUNE: 'Neptune',
            swe.PLUTO: 'Pluto',
            swe.TRUE_NODE: 'North_Node'
        }

        self.zodiac_signs = [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]

        self.aspects_ml = {
            0: ('conjunction', self.ORBS['conjunction']),
            60: ('sextile', self.ORBS['sextile']),
            90: ('square', self.ORBS['square']),
            120: ('trine', self.ORBS['trine']),
            180: ('opposition', self.ORBS['opposition'])
        }

    def get_city_coordinates(self, city_name: str) -> Tuple[float, float, float]:
        """
        Надежное определение координат города.
        Сначала проверяет кэш, затем основные города, затем геокодинг.
        """
        city_lower = city_name.strip().lower()

        # 1. Проверяем кэш
        if city_lower in self.coordinates_cache:
            logger.info(f"Координаты из кэша для: {city_name}")
            return self.coordinates_cache[city_lower]

        # 2. Проверяем основные города России
        if city_lower in self.major_cities:
            coords = self.major_cities[city_lower]
            self.coordinates_cache[city_lower] = coords
            logger.info(f"Координаты из базы основных городов для: {city_name}")
            return coords

        # 3. Используем геокодинг через Nominatim (OpenStreetMap)
        try:
            coords = self._geocode_city(city_name)
            if coords:
                self.coordinates_cache[city_lower] = coords
                logger.info(f"Координаты получены через геокодинг для: {city_name}")
                return coords
        except Exception as e:
            logger.warning(f"Ошибка геокодинга для {city_name}: {e}")

        # 4. Резервный вариант - Москва
        logger.warning(f"Не удалось определить координаты для {city_name}, используем Москву")
        return (55.7558, 37.6173, 156)

    def _geocode_city(self, city_name: str) -> Tuple[float, float, float]:
        """
        Геокодинг города через Nominatim API (OpenStreetMap)
        """
        # Добавляем страну для лучшего определения
        search_query = f"{city_name}, Россия"
        encoded_query = quote(search_query)

        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"

        headers = {
            'User-Agent': 'AstrologyBot/1.0 (leostuchchi@example.com)',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])

                # Определяем высоту (примерно, так как Nominatim не дает точную высоту)
                elevation = self._estimate_elevation(lat, lon)

                logger.info(f"Геокодинг успешен: {city_name} -> {lat}, {lon}, {elevation}м")
                return (lat, lon, elevation)

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса геокодинга для {city_name}: {e}")
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Ошибка парсинга ответа геокодинга для {city_name}: {e}")

        return None

    def _estimate_elevation(self, lat: float, lon: float) -> float:
        """
        Примерная оценка высоты над уровнем моря.
        Для точных данных лучше использовать специализированные API.
        """
        # Простая логика: прибрежные города ~0м, горные ~500м, равнинные ~100-200м
        if 43 <= lat <= 49 and 131 <= lon <= 142:  # Дальний Восток
            return 200
        elif 53 <= lat <= 58 and 48 <= lon <= 56:  # Поволжье
            return 100
        elif 55 <= lat <= 57 and 37 <= lon <= 40:  # Центральная Россия
            return 150
        elif 44 <= lat <= 46 and 38 <= lon <= 40:  # Юг России
            return 50
        elif 51 <= lat <= 53 and 103 <= lon <= 108:  # Байкал
            return 500
        else:
            return 100  # Средняя высота по умолчанию

    def _geocode_fallback(self, city_name: str) -> Tuple[float, float, float]:
        """
        Резервный метод геокодинга через альтернативный сервис
        """
        try:
            # Альтернативный сервис - GeoNames (требует API key)
            # Можно добавить при необходимости
            pass
        except Exception as e:
            logger.warning(f"Резервный геокодинг не сработал: {e}")

        return None

    def add_city_to_cache(self, city_name: str, lat: float, lon: float, elevation: float = 100):
        """
        Ручное добавление города в кэш
        """
        city_lower = city_name.strip().lower()
        self.coordinates_cache[city_lower] = (lat, lon, elevation)
        logger.info(f"Город добавлен в кэш: {city_name}")

    def get_cached_cities(self) -> List[str]:
        """
        Получить список всех закэшированных городов
        """
        return list(self.coordinates_cache.keys())

    # Остальные методы класса остаются без изменений
    def calculate_planet_positions(self, jd_ut: float) -> Dict[str, Dict]:
        positions = {}
        for planet_id, name in self.planets_ml.items():
            try:
                flags = swe.FLG_SWIEPH | swe.FLG_SPEED
                pos, ret_flags = swe.calc_ut(jd_ut, planet_id, flags)
                lon = pos[0] % 360
                sign_index = floor(lon / 30)
                positions[name] = {
                    'longitude': round(lon, 6),
                    'sign': self.zodiac_signs[sign_index],
                    'sign_index': sign_index,
                    'position_in_sign': round(lon % 30, 4),
                    'retrograde': pos[3] < 0,
                    'speed': round(pos[3], 6)
                }
            except Exception as e:
                logger.warning(f"Ошибка расчета для {name}: {e}")
                continue
        return positions

    def calculate_houses_ml(self, jd_ut: float, lat: float, lon: float) -> Dict:
        try:
            hsys = b'P'
            cusps, ascmc = swe.houses(jd_ut, lat, lon, hsys)
            houses = {}
            for i, cusp in enumerate(cusps[:12]):
                cusp_deg = cusp % 360
                sign_index = floor(cusp_deg / 30)
                houses[i + 1] = {
                    'cusp_longitude': round(cusp_deg, 6),
                    'sign': self.zodiac_signs[sign_index],
                    'sign_index': sign_index,
                    'position_in_sign': round(cusp_deg % 30, 4)
                }
            return {
                'houses': houses,
                'ascendant': round(ascmc[0] % 360, 6),
                'midheaven': round(ascmc[1] % 360, 6),
                'house_system': 'Placidus'
            }
        except Exception as e:
            logger.error(f"Ошибка расчета домов: {e}")
            return self._get_default_houses()

    def _get_default_houses(self) -> Dict:
        houses = {}
        for i in range(12):
            houses[i + 1] = {
                'cusp_longitude': round(i * 30.0, 6),
                'sign': self.zodiac_signs[i],
                'sign_index': i,
                'position_in_sign': 0.0
            }
        return {
            'houses': houses,
            'ascendant': 0.0,
            'midheaven': 0.0,
            'house_system': 'Placidus'
        }

    def calculate_aspects_ml(self, planets: Dict, asc: float, mc: float) -> List[Dict]:
        aspects = []
        all_points = {**planets}
        all_points['Ascendant'] = {'longitude': asc}
        all_points['Midheaven'] = {'longitude': mc}
        point_names = list(all_points.keys())
        for i in range(len(point_names)):
            for j in range(i + 1, len(point_names)):
                p1, p2 = point_names[i], point_names[j]
                lon1, lon2 = all_points[p1]['longitude'], all_points[p2]['longitude']
                distance = abs(lon1 - lon2)
                angle = min(distance, 360 - distance)
                for aspect_angle, (aspect_name, orb) in self.aspects_ml.items():
                    if abs(angle - aspect_angle) <= orb:
                        aspects.append({
                            'point1': p1,
                            'point2': p2,
                            'aspect': aspect_name,
                            'exact_angle': aspect_angle,
                            'actual_angle': round(angle, 4),
                            'orb': round(abs(angle - aspect_angle), 4),
                            'strength': 1.0 - (abs(angle - aspect_angle) / orb)
                        })
                        break
        aspects.sort(key=lambda x: x['strength'], reverse=True)
        return aspects

    def get_planet_house_placement(self, planets: Dict, houses: Dict) -> Dict:
        house_placement = {}
        for planet_name, planet_data in planets.items():
            planet_lon = planet_data['longitude']
            for house_num, house_data in houses.items():
                next_house_num = house_num + 1 if house_num < 12 else 1
                next_house_lon = houses[next_house_num]['cusp_longitude']
                current_lon = house_data['cusp_longitude']
                if next_house_lon < current_lon:
                    next_house_lon += 360
                    adjusted_planet_lon = planet_lon + 360 if planet_lon < current_lon else planet_lon
                else:
                    adjusted_planet_lon = planet_lon
                if current_lon <= adjusted_planet_lon < next_house_lon:
                    house_placement[planet_name] = house_num
                    break
            else:
                house_placement[planet_name] = 1
        return house_placement

    def calculate_natal_chart_ml(self, city_name: str, birth_datetime_local: datetime, timezone_str: str) -> Dict[
        str, Any]:
        try:
            lat, lon, elevation = self.get_city_coordinates(city_name)
            local_tz = pytz.timezone(timezone_str)
            birth_local = local_tz.localize(birth_datetime_local)
            birth_utc = birth_local.astimezone(pytz.utc)
            jd_ut = swe.julday(
                birth_utc.year,
                birth_utc.month,
                birth_utc.day,
                birth_utc.hour + birth_utc.minute / 60 + birth_utc.second / 3600
            )
            planets = self.calculate_planet_positions(jd_ut)
            houses_data = self.calculate_houses_ml(jd_ut, lat, lon)
            house_placement = self.get_planet_house_placement(planets, houses_data['houses'])
            aspects = self.calculate_aspects_ml(planets, houses_data['ascendant'], houses_data['midheaven'])
            return {
                'metadata': {
                    'location': {
                        'city': city_name,
                        'lat': round(lat, 4),
                        'lon': round(lon, 4),
                        'elevation': round(elevation, 1)
                    },
                    'datetime': {
                        'local': birth_local.isoformat(),
                        'utc': birth_utc.isoformat(),
                        'jd': round(jd_ut, 6)
                    },
                    'calculation': {
                        'house_system': houses_data['house_system'],
                        'ephemeris': 'DE441'
                    }
                },
                'planets': planets,
                'houses': houses_data['houses'],
                'angles': {
                    'ascendant': {
                        'longitude': houses_data['ascendant'],
                        'sign': self.zodiac_signs[floor(houses_data['ascendant'] / 30)],
                        'sign_index': floor(houses_data['ascendant'] / 30)
                    },
                    'midheaven': {
                        'longitude': houses_data['midheaven'],
                        'sign': self.zodiac_signs[floor(houses_data['midheaven'] / 30)],
                        'sign_index': floor(houses_data['midheaven'] / 30)
                    }
                },
                'placements': house_placement,
                'aspects': aspects,
                'ml_features': {
                    'sign_distribution': self._get_sign_distribution(planets, houses_data),
                    'aspect_patterns': self._get_aspect_patterns(aspects),
                    'element_balance': self._get_element_balance(planets)
                }
            }
        except Exception as e:
            logger.error(f"Ошибка расчета натальной карты: {e}")
            raise

    def _get_sign_distribution(self, planets: Dict, houses_data: Dict) -> Dict[str, int]:
        distribution = {sign: 0 for sign in self.zodiac_signs}
        for planet_data in planets.values():
            distribution[planet_data['sign']] += 1
        return distribution

    def _get_aspect_patterns(self, aspects: List[Dict]) -> Dict[str, int]:
        patterns = {
            'conjunctions': 0,
            'squares': 0,
            'trines': 0,
            'oppositions': 0,
            'sextiles': 0
        }
        for aspect in aspects:
            if aspect['aspect'] in patterns:
                patterns[aspect['aspect']] += 1
        return patterns

    def _get_element_balance(self, planets: Dict) -> Dict[str, int]:
        elements = {
            'fire': ['Aries', 'Leo', 'Sagittarius'],
            'earth': ['Taurus', 'Virgo', 'Capricorn'],
            'air': ['Gemini', 'Libra', 'Aquarius'],
            'water': ['Cancer', 'Scorpio', 'Pisces']
        }
        balance = {element: 0 for element in elements}
        for planet_data in planets.values():
            for element, signs in elements.items():
                if planet_data['sign'] in signs:
                    balance[element] += 1
                    break
        return balance

    def save_ml_chart(self, natal_chart: Dict, filename: str = 'natal_chart_ml.json') -> None:
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(natal_chart, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        logger.info(f"ML-натальная карта сохранена: {filename}")
backend.predictions.py:
from math import floor
import json
from datetime import date, datetime
import swisseph as swe
from sqlalchemy.future import select

from backend.database import async_session, NatalPredictions


class AstroPredictor:
    def __init__(self, natal_chart):
        self.natal_chart = natal_chart
        self.planets_ml = {
            swe.SUN: 'Sun', swe.MOON: 'Moon', swe.MERCURY: 'Mercury',
            swe.VENUS: 'Venus', swe.MARS: 'Mars', swe.JUPITER: 'Jupiter',
            swe.SATURN: 'Saturn', swe.URANUS: 'Uranus',
            swe.NEPTUNE: 'Neptune', swe.PLUTO: 'Pluto'
        }
        # Русские названия для пользователя
        self.planet_names_ru = {
            'Sun': 'Солнце', 'Moon': 'Луна', 'Mercury': 'Меркурий',
            'Venus': 'Венера', 'Mars': 'Марс', 'Jupiter': 'Юпитер',
            'Saturn': 'Сатурн', 'Uranus': 'Уран', 'Neptune': 'Нептун', 'Pluto': 'Плутон'
        }
        self.sign_names_ru = {
            'Aries': 'Овен', 'Taurus': 'Телец', 'Gemini': 'Близнецы',
            'Cancer': 'Рак', 'Leo': 'Лев', 'Virgo': 'Дева',
            'Libra': 'Весы', 'Scorpio': 'Скорпион', 'Sagittarius': 'Стрелец',
            'Capricorn': 'Козерог', 'Aquarius': 'Водолей', 'Pisces': 'Рыбы'
        }
        self.planet_names_to_ids = {v: k for k, v in self.planets_ml.items()}

    def calculate_transits(self, target_date):
        jd_target = swe.julday(target_date.year, target_date.month, target_date.day, 12.0)
        transits = {}
        for planet_id, name in self.planets_ml.items():
            pos, _ = swe.calc_ut(jd_target, planet_id, swe.FLG_SWIEPH)
            lon = pos[0] % 360
            transits[name] = {
                'longitude': lon,
                'sign': self.get_sign_from_longitude(lon),
                'position_in_sign': lon % 30,
                'retrograde': pos[3] < 0
            }
        return transits

    def get_sign_from_longitude(self, longitude):
        signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                 "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
        return signs[floor(longitude / 30)]

    def analyze_aspects(self, transits, natal_positions):
        aspects = []
        for t_planet, t_data in transits.items():
            for n_planet, n_data in natal_positions.items():
                if t_planet == n_planet:
                    continue
                t_lon = t_data['longitude']
                n_lon = n_data['longitude']
                distance = abs(t_lon - n_lon)
                angle = min(distance, 360 - distance)
                aspect_info = self.check_aspect(angle)
                if aspect_info:
                    aspects.append({
                        'transit_planet': t_planet,
                        'natal_planet': n_planet,
                        'aspect': aspect_info[0],
                        'exact_angle': aspect_info[1],
                        'actual_angle': angle,
                        'orb': abs(angle - aspect_info[1]),
                        'strength': 1.0 - (abs(angle - aspect_info[1]) / aspect_info[2])
                    })
        aspects.sort(key=lambda x: x['strength'], reverse=True)
        return aspects

    def check_aspect(self, angle):
        aspects = {
            0: ('conjunction', 0, 8),
            60: ('sextile', 60, 6),
            90: ('square', 90, 8),
            120: ('trine', 120, 8),
            180: ('opposition', 180, 8)
        }
        for aspect_angle, (name, exact, orb) in aspects.items():
            if abs(angle - aspect_angle) <= orb:
                return (name, exact, orb)
        return None

    def get_planet_influence(self, planet_name):
        """Определяет сферу влияния планеты на основе астрологических принципов"""
        influences = {
            'Sun': 'личную энергию, творчество, самореализацию',
            'Moon': 'эмоции, интуицию, домашние дела',
            'Mercury': 'общение, обучение, документы',
            'Venus': 'отношения, финансы, искусство',
            'Mars': 'действия, инициативу, спорт',
            'Jupiter': 'расширение, возможности, путешествия',
            'Saturn': 'ответственность, карьеру, долгосрочные планы',
            'Uranus': 'изменения, инновации, неожиданные события',
            'Neptune': 'интуицию, творчество, духовность',
            'Pluto': 'трансформацию, глубокие изменения'
        }
        return influences.get(planet_name, 'личное развитие')

    def get_aspect_meaning(self, aspect_type, strength):
        """Получает значение аспекта в зависимости от силы на основе астрологических принципов"""
        if aspect_type == 'conjunction':
            if strength > 0.7:
                return "мощное соединение - время начинаний"
            else:
                return "соединение - новые возможности"
        elif aspect_type == 'opposition':
            if strength > 0.7:
                return "сильная оппозиция - важные решения"
            else:
                return "оппозиция - требует баланса"
        elif aspect_type == 'square':
            if strength > 0.7:
                return "напряженный квадрат - преодоление препятствий"
            else:
                return "квадрат - вызовы для роста"
        elif aspect_type == 'trine':
            if strength > 0.7:
                return "гармоничный трин - благоприятное время"
            else:
                return "трин - поддержка и удача"
        elif aspect_type == 'sextile':
            if strength > 0.7:
                return "благоприятный секстиль - хорошие возможности"
            else:
                return "секстиль - шансы для развития"
        return "влияние на вашу энергию"

    def generate_personal_recommendations(self, aspects, transits):
        """Генерация персонализированных рекомендаций на основе РАСЧЕТОВ аспектов"""
        recommendations = []
        warnings = []

        # Анализируем самые сильные аспекты (топ-3)
        strong_aspects = [a for a in aspects if a['strength'] > 0.6][:3]

        for aspect in strong_aspects:
            transit_planet_ru = self.planet_names_ru.get(aspect['transit_planet'], aspect['transit_planet'])
            natal_planet_ru = self.planet_names_ru.get(aspect['natal_planet'], aspect['natal_planet'])
            influence_area = self.get_planet_influence(aspect['natal_planet'])
            aspect_meaning = self.get_aspect_meaning(aspect['aspect'], aspect['strength'])

            recommendation = f"{transit_planet_ru} {aspect_meaning} в сфере {influence_area}"

            if aspect['aspect'] in ['trine', 'sextile', 'conjunction']:
                # Благоприятные аспекты на основе РАСЧЕТОВ
                if aspect['aspect'] == 'conjunction':
                    actions = {
                        'Sun': 'начинайте новые проекты, проявляйте инициативу',
                        'Moon': 'доверяйте интуиции, займитесь домом',
                        'Mercury': 'общайтесь, учитесь, подписывайте документы',
                        'Venus': 'укрепляйте отношения, занимайтесь творчеством',
                        'Mars': 'действуйте решительно, занимайтесь спортом',
                        'Jupiter': 'расширяйте горизонты, путешествуйте',
                        'Saturn': 'стройте долгосрочные планы, берите ответственность',
                        'Uranus': 'экспериментируйте, будьте открыты новому',
                        'Neptune': 'развивайте интуицию, занимайтесь творчеством',
                        'Pluto': 'трансформируйте старые привычки'
                    }
                elif aspect['aspect'] in ['trine', 'sextile']:
                    actions = {
                        'Sun': 'используйте свою энергию для творчества',
                        'Moon': 'положитесь на внутренние ощущения',
                        'Mercury': 'эффективно общайтесь и договаривайтесь',
                        'Venus': 'гармонизируйте отношения и финансы',
                        'Mars': 'реализуйте планы с энтузиазмом',
                        'Jupiter': 'используйте расширяющиеся возможности',
                        'Saturn': 'стройте прочный фундамент',
                        'Uranus': 'внедряйте инновационные идеи',
                        'Neptune': 'развивайте духовные практики',
                        'Pluto': 'глубоко трансформируйтесь'
                    }

                action = actions.get(aspect['natal_planet'], 'используйте эту энергию для роста')
                recommendations.append(f"{recommendation}. {action} (сила аспекта: {aspect['strength']:.2f})")

            else:
                # Сложные аспекты (квадрат, оппозиция) на основе РАСЧЕТОВ
                cautions = {
                    'Sun': 'избегайте конфликтов, будьте дипломатичны',
                    'Moon': 'контролируйте эмоции, избегайте импульсивности',
                    'Mercury': 'проверяйте информацию, избегайте споров',
                    'Venus': 'будьте осторожны в отношениях и финансах',
                    'Mars': 'избегайте рисков, действуйте обдуманно',
                    'Jupiter': 'не переоценивайте возможности',
                    'Saturn': 'не избегайте ответственности, но и не перегружайтесь',
                    'Uranus': 'будьте готовы к неожиданностям',
                    'Neptune': 'различайте иллюзии и реальность',
                    'Pluto': 'избегайте манипуляций и давления'
                }

                caution = cautions.get(aspect['natal_planet'], 'будьте внимательны и осторожны')
                warnings.append(f"{recommendation}. {caution} (сила аспекта: {aspect['strength']:.2f})")

        # Добавляем информацию о ретроградных планетах на основе РАСЧЕТОВ
        retrograde_planets = [p for p, data in transits.items() if data.get('retrograde')]
        if retrograde_planets:
            retro_names = [self.planet_names_ru.get(p, p) for p in retrograde_planets]
            if len(retro_names) > 0:
                warnings.append(f"Ретроградные {', '.join(retro_names)} - время пересмотра и анализа")

        # Если аспектов мало, добавляем рекомендации на основе общей картины РАСЧЕТОВ
        if not recommendations and not warnings:
            total_aspects = len(aspects)
            if total_aspects > 0:
                avg_strength = sum(a['strength'] for a in aspects) / total_aspects
                recommendations.append(
                    f"Наблюдается {total_aspects} аспектов со средней силой {avg_strength:.2f} - следите за изменениями в соответствующих сферах")
            else:
                # Если аспектов нет вообще - это тоже результат расчета
                recommendations.append(
                    "Сегодня минимальная астрологическая активность - хороший день для рутинных дел и планирования")

        return recommendations[:4], warnings[:3]  # Ограничиваем количество

    def analyze_natal_elements(self):
        """Анализирует элементный баланс натальной карты на основе РАСЧЕТОВ"""
        if 'ml_features' in self.natal_chart and 'element_balance' in self.natal_chart['ml_features']:
            return self.natal_chart['ml_features']['element_balance']
        return None

    def get_element_recommendation(self, elements):
        """Рекомендации на основе РАСЧЕТОВ элементного баланса"""
        if not elements:
            return None

        max_element = max(elements.items(), key=lambda x: x[1])
        element_value = max_element[1]

        recommendations = {
            'fire': f"Доминирует огонь ({element_value} планет) - используйте свою энергию и инициативу для новых начинаний",
            'earth': f"Доминирует земля ({element_value} планет) - сосредоточьтесь на практических задачах и стабильности",
            'air': f"Доминирует воздух ({element_value} планет) - развивайте общение, обучение и интеллектуальную деятельность",
            'water': f"Доминирует вода ({element_value} планет) - доверяйте интуиции и развивайте эмоциональную чувствительность"
        }

        return recommendations.get(max_element[0])

    def generate_prediction(self, target_date):
        """Основной метод генерации предсказания на основе РАСЧЕТОВ"""
        try:
            # Рассчитываем транзиты
            transits = self.calculate_transits(target_date)

            # Получаем натальные позиции
            natal_positions = {}
            for name, data in self.natal_chart['planets'].items():
                if name in self.planets_ml.values():
                    natal_positions[name] = {
                        'longitude': data['longitude'],
                        'sign': data['sign'],
                        'position_in_sign': data['position_in_sign']
                    }

            # Добавляем углы карты
            if 'angles' in self.natal_chart:
                natal_positions['Ascendant'] = {
                    'longitude': self.natal_chart['angles']['ascendant']['longitude'],
                    'sign': self.natal_chart['angles']['ascendant']['sign'],
                    'position_in_sign': self.natal_chart['angles']['ascendant']['longitude'] % 30
                }

            # Анализируем аспекты
            aspects = self.analyze_aspects(transits, natal_positions)

            # Генерируем персонализированные рекомендации и предостережения на основе РАСЧЕТОВ
            recommendations, warnings = self.generate_personal_recommendations(aspects, transits)

            # Добавляем рекомендации на основе элементного баланса если есть
            element_balance = self.analyze_natal_elements()
            if element_balance:
                element_recommendation = self.get_element_recommendation(element_balance)
                if element_recommendation and len(recommendations) < 4:
                    recommendations.append(element_recommendation)

            return {
                'prediction_date': target_date.strftime('%Y-%m-%d'),
                'significant_aspects': aspects[:5],
                'recommendations': recommendations,
                'warnings': warnings,
                'transits_count': len(transits),
                'aspects_count': len(aspects),
                'strong_aspects_count': len([a for a in aspects if a['strength'] > 0.7])
            }

        except Exception as e:
            # В случае ошибки возвращаем пустое предсказание с информацией об ошибке
            return {
                'prediction_date': target_date.strftime('%Y-%m-%d'),
                'significant_aspects': [],
                'recommendations': [f"Ошибка расчета: {str(e)} - обратитесь к администратору"],
                'warnings': ["Временные технические трудности при расчете аспектов"],
                'transits_count': 0,
                'aspects_count': 0,
                'strong_aspects_count': 0,
                'calculation_error': True
            }

    async def save_prediction_to_db(self, telegram_id: int, prediction_date: date):
        """Сохранение предсказания в базу данных"""
        prediction = self.generate_prediction(prediction_date)
        async with async_session() as session:
            result = await session.execute(
                select(NatalPredictions).where(NatalPredictions.telegram_id == telegram_id)
            )
            existing_record = result.scalar_one_or_none()

            if existing_record:
                existing_record.predictions = prediction
                existing_record.updated_at = datetime.utcnow()
            else:
                new_record = NatalPredictions(
                    telegram_id=telegram_id,
                    predictions=prediction,
                    assistant_data={},
                )
                session.add(new_record)

            await session.commit()
        return prediction
backend.prediction_services.py:
from backend.database import async_session, NatalPredictions
from backend.predictions import AstroPredictor
from backend.chart_services import get_user_natal_chart
from backend.biorhythm_services import calculate_and_save_biorhythms
from sqlalchemy.future import select
from sqlalchemy import func
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class PredictionCombiner:
    """Класс для объединения астрологических предсказаний и биоритмов"""

    def __init__(self):
        pass

    def combine_recommendations(self, astro_prediction: dict, biorhythm_data: dict) -> list:
        """Объединение рекомендаций из астрологии и биоритмов на основе РАСЧЕТОВ"""

        # Берем рекомендации из обоих источников
        astro_recommendations = astro_prediction.get('recommendations', [])
        biorhythm_recommendations = biorhythm_data.get('recommendations', [])

        # Объединяем рекомендации
        all_recommendations = astro_recommendations + biorhythm_recommendations

        # Сортируем по приоритету на основе РАСЧЕТОВ
        priority_recommendations = self._prioritize_recommendations(all_recommendations)

        return priority_recommendations[:8]  # Не более 8 рекомендаций

    def _prioritize_recommendations(self, recommendations: list) -> list:
        """Приоритизация рекомендаций на основе РАСЧЕТОВ"""
        high_priority = []
        medium_priority = []
        low_priority = []

        for rec in recommendations:
            rec_lower = rec.lower()

            # Высокий приоритет - предостережения и критические дни на основе РАСЧЕТОВ
            if any(word in rec_lower for word in
                   ['осторожн', 'избегай', 'опасн', 'критич', 'не рискуй', 'береги', 'ретроградн']):
                high_priority.append(rec)
            # Средний приоритет - активные действия на основе РАСЧЕТОВ
            elif any(word in rec_lower for word in ['идеальн', 'отличн', 'благоприятн', 'используй', 'высок', 'пик']):
                medium_priority.append(rec)
            # Низкий приоритет - информационные рекомендации
            else:
                low_priority.append(rec)

        return high_priority + medium_priority + low_priority

    def generate_energy_analysis(self, astro_prediction: dict, biorhythm_data: dict) -> str:
        """Анализ энергетического состояния на основе РАСЧЕТОВ обоих методов"""

        # Данные из биоритмов
        energy_level = biorhythm_data.get('overall_energy', {}).get('level', 'средний')
        energy_percentage = biorhythm_data.get('overall_energy', {}).get('percentage', 50)

        # Данные из астрологии
        aspects = astro_prediction.get('significant_aspects', [])
        strong_aspects = [a for a in aspects if a.get('strength', 0) > 0.7]
        challenging_aspects = [a for a in strong_aspects if a.get('aspect') in ['square', 'opposition']]
        harmonious_aspects = [a for a in strong_aspects if a.get('aspect') in ['trine', 'sextile', 'conjunction']]

        # Формируем анализ на основе РАСЧЕТОВ
        analysis_parts = []

        # Анализ энергии из биоритмов
        analysis_parts.append(f"⚡ Уровень энергии: {energy_level} ({energy_percentage:.1f}%)")

        # Анализ аспектов из астрологии
        if challenging_aspects:
            analysis_parts.append(f"🎯 Сложных аспектов: {len(challenging_aspects)}")

        if harmonious_aspects:
            analysis_parts.append(f"🌟 Гармоничных аспектов: {len(harmonious_aspects)}")

        # Общий вывод на основе РАСЧЕТОВ
        if energy_percentage > 70 and len(challenging_aspects) == 0:
            analysis_parts.append("✅ Идеальный день для активных действий")
        elif energy_percentage < 30 and len(challenging_aspects) > 2:
            analysis_parts.append("⚠️ Сохраняйте спокойствие, избегайте нагрузок")
        elif len(harmonious_aspects) > len(challenging_aspects):
            analysis_parts.append("📊 Преобладают гармоничные влияния")
        else:
            analysis_parts.append("📈 Сбалансированный энергетический профиль")

        return " | ".join(analysis_parts)

    def create_daily_schedule(self, biorhythm_data: dict) -> list:
        """Создание рекомендуемого расписания дня на основе РАСЧЕТОВ биоритмов"""

        cycles = biorhythm_data.get('cycles', {})
        schedule = []

        # Утренние рекомендации на основе РАСЧЕТОВ интеллектуального цикла
        morning_rec = "🌅 Утро: "
        intellectual_value = cycles.get('intellectual', {}).get('value', 0)
        if intellectual_value > 0.3:
            morning_rec += f"планирование и анализ (интеллектуальный цикл: {intellectual_value:.2f})"
        else:
            morning_rec += f"легкая разминка и рутина (интеллектуальный цикл: {intellectual_value:.2f})"
        schedule.append(morning_rec)

        # Дневные рекомендации на основе РАСЧЕТОВ физического цикла
        day_rec = "🌞 День: "
        physical_value = cycles.get('physical', {}).get('value', 0)
        if physical_value > 0.5:
            day_rec += f"активная работа и движение (физический цикл: {physical_value:.2f})"
        elif physical_value > 0:
            day_rec += f"умеренная активность (физический цикл: {physical_value:.2f})"
        else:
            day_rec += f"спокойная деятельность (физический цикл: {physical_value:.2f})"
        schedule.append(day_rec)

        # Вечерние рекомендации на основе РАСЧЕТОВ эмоционального цикла
        evening_rec = "🌙 Вечер: "
        emotional_value = cycles.get('emotional', {}).get('value', 0)
        if emotional_value > 0.4:
            evening_rec += f"общение и творчество (эмоциональный цикл: {emotional_value:.2f})"
        else:
            evening_rec += f"отдых и уединение (эмоциональный цикл: {emotional_value:.2f})"
        schedule.append(evening_rec)

        return schedule

    def _extract_critical_notes(self, astro_prediction: dict, biorhythm_data: dict) -> list:
        """Извлечение критических замечаний на основе РАСЧЕТОВ обоих источников"""
        critical_notes = []

        # Критические дни из биоритмов на основе РАСЧЕТОВ
        critical_days = biorhythm_data.get('critical_days', [])
        if critical_days:
            for day in critical_days:
                critical_notes.append(f"⚠️ {day.get('description', 'Критический день по биоритмам')}")

        # Сложные аспекты из астрологии на основе РАСЧЕТОВ
        aspects = astro_prediction.get('significant_aspects', [])
        challenging_aspects = [a for a in aspects if
                               a.get('aspect') in ['square', 'opposition'] and a.get('strength', 0) > 0.7]

        for aspect in challenging_aspects[:2]:  # Не более 2 самых сильных
            planet1 = aspect.get('transit_planet', '')
            planet2 = aspect.get('natal_planet', '')
            aspect_type = aspect.get('aspect', '')
            strength = aspect.get('strength', 0)

            planet1_ru = self._get_planet_name_ru(planet1)
            planet2_ru = self._get_planet_name_ru(planet2)

            if aspect_type == 'square':
                critical_notes.append(f"🔺 Напряженный аспект: {planet1_ru} - {planet2_ru} (сила: {strength:.2f})")
            elif aspect_type == 'opposition':
                critical_notes.append(f"⚖️ Сложный выбор: {planet1_ru} - {planet2_ru} (сила: {strength:.2f})")

        # Предостережения из астрологии
        warnings = astro_prediction.get('warnings', [])
        critical_notes.extend(warnings[:2])  # Не более 2 предостережений

        return critical_notes[:4]  # Не более 4 критических заметок

    def _get_planet_name_ru(self, planet_name: str) -> str:
        """Получение русского названия планеты"""
        planet_names_ru = {
            'Sun': 'Солнце', 'Moon': 'Луна', 'Mercury': 'Меркурий',
            'Venus': 'Венера', 'Mars': 'Марс', 'Jupiter': 'Юпитер',
            'Saturn': 'Сатурн', 'Uranus': 'Уран', 'Neptune': 'Нептун', 'Pluto': 'Плутон'
        }
        return planet_names_ru.get(planet_name, planet_name)


async def generate_and_save_prediction(telegram_id: int, target_date: date):
    """Генерация и сохранение предсказания с биоритмами на основе РАСЧЕТОВ"""
    try:
        logger.info(f"🔮 Генерация предсказания для пользователя {telegram_id} на {target_date}")

        # Получаем натальную карту пользователя
        natal_data = await get_user_natal_chart(telegram_id)
        if not natal_data:
            logger.warning(f"⚠️ Натальная карта не найдена для пользователя {telegram_id}")
            raise ValueError("Натальная карта не найдена. Сначала создайте натальную карту с помощью /start")

        logger.info(f"✅ Натальная карта найдена для {telegram_id}")

        # Рассчитываем биоритмы на основе РАСЧЕТОВ
        biorhythm_data = await calculate_and_save_biorhythms(telegram_id, target_date)
        logger.info(f"✅ Биоритмы рассчитаны для {telegram_id}")

        # Генерируем астрологическое предсказание на основе РАСЧЕТОВ
        predictor = AstroPredictor(natal_data)
        astro_prediction = predictor.generate_prediction(target_date)
        logger.info(f"✅ Астрологическое предсказание сгенерировано для {telegram_id}")

        # Объединяем предсказания на основе РАСЧЕТОВ
        combiner = PredictionCombiner()
        combined_recommendations = combiner.combine_recommendations(astro_prediction, biorhythm_data)
        energy_analysis = combiner.generate_energy_analysis(astro_prediction, biorhythm_data)
        daily_schedule = combiner.create_daily_schedule(biorhythm_data)
        critical_notes = combiner._extract_critical_notes(astro_prediction, biorhythm_data)


        # Создаем финальное предсказание на основе РАСЧЕТОВ
        final_prediction = {
            'prediction_date': target_date.isoformat(),
            'energy_analysis': energy_analysis,
            'biorhythms_summary': {
                'overall_energy': biorhythm_data.get('overall_energy', {}),
                'physical_cycle': biorhythm_data.get('cycles', {}).get('physical', {}),
                'emotional_cycle': biorhythm_data.get('cycles', {}).get('emotional', {}),
                'intellectual_cycle': biorhythm_data.get('cycles', {}).get('intellectual', {}),
                'critical_days_count': len(biorhythm_data.get('critical_days', [])),
                'peak_days_count': len(biorhythm_data.get('peak_days', []))
            },
            'astro_summary': {
                'significant_aspects_count': len(astro_prediction.get('significant_aspects', [])),
                'strong_aspects_count': astro_prediction.get('strong_aspects_count', 0),
                'transits_count': astro_prediction.get('transits_count', 0),
                'key_aspects': astro_prediction.get('significant_aspects', [])[:3]
            },
            'combined_recommendations': combined_recommendations,
            'daily_schedule': daily_schedule,
            'critical_notes': critical_notes,

            # Полные данные для детального анализа
            'full_astro_prediction': astro_prediction,
            'full_biorhythm_data': biorhythm_data,

            # Мета-информация о расчетах
            'calculation_metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'data_sources': ['astrology', 'biorhythms'],
                'calculation_methods': ['swiss_ephemeris', 'sine_wave_analysis']
            }
        }

        logger.info(f"✅ Комбинированное предсказание создано для {telegram_id}")

        # Сохраняем предсказание в БД
        async with async_session() as session:
            result = await session.execute(
                select(NatalPredictions).where(NatalPredictions.telegram_id == telegram_id)
            )
            existing_record = result.scalar_one_or_none()

            if existing_record:
                # Обновляем существующую запись
                existing_record.predictions = final_prediction
                existing_record.updated_at = func.now()
                logger.info(f"📝 Обновлено существующее предсказание для {telegram_id}")
            else:
                # Создаем новую запись
                new_record = NatalPredictions(
                    telegram_id=telegram_id,
                    predictions=final_prediction,
                    assistant_data={},
                )
                session.add(new_record)
                logger.info(f"🆕 Создано новое предсказание для {telegram_id}")

            await session.commit()
            logger.info(f"💾 Предсказание успешно сохранено в БД для {telegram_id}")

        return final_prediction

    except ValueError as e:
        logger.warning(f"❌ Ошибка валидации для {telegram_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка при генерации предсказания для {telegram_id}: {e}")
        raise Exception(f"Не удалось сгенерировать предсказание на основе расчетов: {str(e)}")


async def get_user_predictions(telegram_id: int):
    """Получение предсказаний пользователя"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(NatalPredictions).where(NatalPredictions.telegram_id == telegram_id)
            )
            predictions = result.scalar_one_or_none()

            if predictions:
                return predictions.predictions
            return None

    except Exception as e:
        logger.error(f"❌ Ошибка при получении предсказаний {telegram_id}: {e}")
        return None


async def get_todays_prediction(telegram_id: int):
    """Получение предсказания на сегодня"""
    try:
        today = datetime.now().date()

        # Получаем сохраненное предсказание
        predictions = await get_user_predictions(telegram_id)

        if predictions and predictions.get('prediction_date') == today.isoformat():
            logger.info(f"✅ Использовано сохраненное предсказание для {telegram_id}")
            return predictions

        # Если предсказания на сегодня нет, генерируем новое на основе РАСЧЕТОВ
        logger.info(f"🔄 Генерация нового предсказания для {telegram_id}")
        return await generate_and_save_prediction(telegram_id, today)

    except Exception as e:
        logger.error(f"❌ Ошибка при получении сегодняшнего предсказания {telegram_id}: {e}")
        return None


async def format_prediction_for_display(prediction: dict) -> str:
    """Форматирование предсказания для отображения в боте на основе РАСЧЕТОВ"""
    if not prediction:
        return "❌ Не удалось получить предсказание на основе расчетов"

    try:
        lines = []
        prediction_date = prediction.get('prediction_date', 'сегодня')
        lines.append(f"🔮 **Ваше предсказание на {prediction_date}**")
        lines.append("")

        # Анализ энергии на основе РАСЧЕТОВ
        energy_analysis = prediction.get('energy_analysis', '')
        if energy_analysis:
            lines.append(f"⚡ {energy_analysis}")
            lines.append("")

        # Биоритмы на основе РАСЧЕТОВ
        biorhythms = prediction.get('biorhythms_summary', {})
        if biorhythms:
            overall_energy = biorhythms.get('overall_energy', {})
            lines.append(
                f"📊 **Биоритмы:** {overall_energy.get('level', 'средний').title()} уровень энергии ({overall_energy.get('percentage', 0):.1f}%)")

            physical = biorhythms.get('physical_cycle', {})
            emotional = biorhythms.get('emotional_cycle', {})
            intellectual = biorhythms.get('intellectual_cycle', {})

            lines.append(
                f"💪 Физический: {physical.get('phase', 'нейтральная')} ({physical.get('percentage', 0):.1f}%) - {physical.get('trend', 'стабильно')}")
            lines.append(
                f"😊 Эмоциональный: {emotional.get('phase', 'нейтральная')} ({emotional.get('percentage', 0):.1f}%) - {emotional.get('trend', 'стабильно')}")
            lines.append(
                f"🧠 Интеллектуальный: {intellectual.get('phase', 'нейтральная')} ({intellectual.get('percentage', 0):.1f}%) - {intellectual.get('trend', 'стабильно')}")
            lines.append("")

        # Астрологическая сводка на основе РАСЧЕТОВ
        astro_summary = prediction.get('astro_summary', {})
        if astro_summary:
            lines.append(
                f"🌟 **Астрология:** {astro_summary.get('significant_aspects_count', 0)} аспектов, {astro_summary.get('strong_aspects_count', 0)} сильных")
            lines.append("")

        # Расписание дня на основе РАСЧЕТОВ биоритмов
        schedule = prediction.get('daily_schedule', [])
        if schedule:
            lines.append("🕒 **Рекомендуемое расписание на основе биоритмов:**")
            for item in schedule:
                lines.append(f"   {item}")
            lines.append("")

        # Рекомендации на основе РАСЧЕТОВ
        recommendations = prediction.get('combined_recommendations', [])
        if recommendations:
            lines.append("💫 **Рекомендации на день (на основе расчетов):**")
            for i, rec in enumerate(recommendations[:6], 1):  # Не более 6 рекомендаций
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Критические заметки на основе РАСЧЕТОВ
        critical_notes = prediction.get('critical_notes', [])
        if critical_notes:
            lines.append("⚠️ **Обратите внимание (на основе расчетов):**")
            for note in critical_notes[:3]:  # Не более 3 заметок
                lines.append(f"   • {note}")
            lines.append("")

        # Информация о расчетах
        lines.append("📈 *Все рекомендации основаны на математических расчетах:*")
        lines.append("   • Астрологические транзиты и аспекты")
        lines.append("   • Биоритмы (физический, эмоциональный, интеллектуальный циклы)")
        lines.append("   • Статистический анализ влияний")

        # ✅ ВАЖНО: Возвращаем объединенную строку, а не список
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"❌ Ошибка форматирования предсказания: {e}")
        return "❌ Произошла ошибка при формировании предсказания на основе расчетов"


async def get_prediction_statistics(telegram_id: int) -> dict:
    """Получение статистики предсказаний пользователя"""
    try:
        prediction = await get_user_predictions(telegram_id)
        if not prediction:
            return {}

        return {
            'last_calculation_date': prediction.get('prediction_date'),
            'biorhythm_energy': prediction.get('biorhythms_summary', {}).get('overall_energy', {}).get('percentage', 0),
            'astro_aspects_count': prediction.get('astro_summary', {}).get('significant_aspects_count', 0),
            'recommendations_count': len(prediction.get('combined_recommendations', [])),
            'critical_notes_count': len(prediction.get('critical_notes', []))
        }

    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики для {telegram_id}: {e}")
        return {}


async def validate_prediction_data(telegram_id: int) -> bool:
    """Проверка корректности данных предсказания"""
    try:
        prediction = await get_user_predictions(telegram_id)
        if not prediction:
            return False

        # Проверяем наличие обязательных полей
        required_fields = ['prediction_date', 'energy_analysis', 'combined_recommendations']
        for field in required_fields:
            if field not in prediction or not prediction[field]:
                return False

        # Проверяем что рекомендации не пустые
        if not prediction.get('combined_recommendations'):
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка валидации данных предсказания для {telegram_id}: {e}")
        return False


async def cleanup_old_predictions():
    """Очистка устаревших предсказаний (для администрирования)"""
    try:
        # В текущей структуре у нас только одно предсказание на пользователя
        # Эта функция может быть использована для будущих расширений
        logger.info("🔄 Очистка устаревших предсказаний не требуется в текущей структуре")
        return 0

    except Exception as e:
        logger.error(f"❌ Ошибка при очистке предсказаний: {e}")
        return 0
backend.psyho_matrix.py:
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PsyhoMatrixCalculator:
    def __init__(self):
        pass

    def calculate_matrix(self, birth_date: datetime.date):
        """Расчет психоматрицы по дате рождения (нумерология Пифагора)"""
        day = birth_date.day
        month = birth_date.month
        year = birth_date.year

        # Преобразуем дату в строку для расчетов
        date_str = f"{day:02d}{month:02d}{year}"

        # Первое число - сумма всех цифр даты
        first_number = sum(int(d) for d in date_str)

        # Второе число - сумма цифр первого числа
        second_number = sum(int(d) for d in str(first_number))

        # Третье число - первое число минус удвоенная первая цифра дня рождения
        first_digit_of_day = day // 10
        third_number = first_number - 2 * first_digit_of_day

        # Четвертое число - сумма цифр третьего числа
        fourth_number = sum(int(d) for d in str(third_number))

        # Строим матрицу 3x3 по методу Пифагора
        matrix_numbers = self._build_pythagoras_matrix(day, month, year)

        # Анализируем характеристики на основе РАСЧЕТОВ
        matrix_data = {
            'basic_numbers': {
                'first': first_number,
                'second': second_number,
                'third': third_number,
                'fourth': fourth_number
            },
            'pythagoras_matrix': matrix_numbers,
            'characteristics': self._calculate_characteristics(matrix_numbers),
            'energy_level': self._analyze_energy(first_number),
            'life_purpose': self._analyze_life_purpose(fourth_number),
            'talents': self._analyze_talents(matrix_numbers),
            'calculated_at': datetime.now().isoformat()
        }

        return matrix_data

    def _build_pythagoras_matrix(self, day: int, month: int, year: int):
        """Построение психоматрицы Пифагора 3x3"""
        # Собираем все цифры даты рождения
        all_digits = []
        all_digits.extend([int(d) for d in str(day)])
        all_digits.extend([int(d) for d in str(month)])
        all_digits.extend([int(d) for d in str(year)])

        # Считаем количество каждой цифры от 1 до 9
        matrix = {}
        for i in range(1, 10):
            matrix[str(i)] = all_digits.count(i)

        return matrix

    def _calculate_characteristics(self, matrix):
        """Расчет характеристик на основе матрицы"""
        characteristics = {
            'character': self._analyze_character(matrix),
            'energy': self._analyze_energy_level(matrix),
            'interest': self._analyze_interest(matrix),
            'health': self._analyze_health(matrix),
            'logic': self._analyze_logic(matrix),
            'labor': self._analyze_labor(matrix),
            'luck': self._analyze_luck(matrix),
            'duty': self._analyze_duty(matrix),
            'memory': self._analyze_memory(matrix)
        }
        return characteristics

    def _analyze_character(self, matrix):
        """Анализ характера по цифре 1 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_1 = matrix.get('1', 0)
        if count_1 == 1:
            return "Уравновешенный характер"
        elif count_1 == 2:
            return "Сильный характер, лидерские качества"
        elif count_1 >= 3:
            return "Очень сильный характер, возможна жесткость"
        else:
            return "Мягкий характер, нуждается в поддержке"

    def _analyze_energy_level(self, matrix):
        """Анализ энергии по цифре 2 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_2 = matrix.get('2', 0)
        if count_2 == 1:
            return "Средний уровень энергии"
        elif count_2 == 2:
            return "Высокая энергия, экстрасенсорные способности"
        elif count_2 >= 3:
            return "Очень высокая энергия, нужно учиться управлять"
        else:
            return "Низкая энергия, берегите силы"

    def _analyze_interest(self, matrix):
        """Анализ интересов по цифре 3 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_3 = matrix.get('3', 0)
        if count_3 == 1:
            return "Разносторонние интересы"
        elif count_3 >= 2:
            return "Глубокие интересы в точных науках"
        else:
            return "Творческие интересы"

    def _analyze_health(self, matrix):
        """Анализ здоровья по цифре 4 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_4 = matrix.get('4', 0)
        if count_4 == 1:
            return "Хорошее здоровье"
        elif count_4 >= 2:
            return "Отличное здоровье, выносливость"
        else:
            return "Внимание к здоровью"

    def _analyze_logic(self, matrix):
        """Анализ логики по цифре 5 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_5 = matrix.get('5', 0)
        if count_5 == 1:
            return "Практическая логика"
        elif count_5 >= 2:
            return "Сильная интуиция, предвидение"
        else:
            return "Образное мышление"

    def _analyze_labor(self, matrix):
        """Анализ трудолюбия по цифре 6 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_6 = matrix.get('6', 0)
        if count_6 == 1:
            return "Физический труд приносит удовольствие"
        elif count_6 >= 2:
            return "Трудоголик, любит ручной труд"
        else:
            return "Интеллектуальный труд"

    def _analyze_luck(self, matrix):
        """Анализ удачи по цифре 7 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_7 = matrix.get('7', 0)
        if count_7 == 1:
            return "Удача в мелочах"
        elif count_7 >= 2:
            return "Везение, ангел-хранитель"
        else:
            return "Нужно прилагать усилия"

    def _analyze_duty(self, matrix):
        """Анализ чувства долга по цифре 8 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_8 = matrix.get('8', 0)
        if count_8 == 1:
            return "Ответственность, надежность"
        elif count_8 >= 2:
            return "Сильное чувство долга"
        else:
            return "Свобода важнее обязательств"

    def _analyze_memory(self, matrix):
        """Анализ памяти по цифре 9 - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        count_9 = matrix.get('9', 0)
        if count_9 == 1:
            return "Хорошая память"
        elif count_9 >= 2:
            return "Отличная память, умственные способности"
        else:
            return "Практическая память"

    def _analyze_energy(self, first_number):
        """Анализ общего уровня энергии - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        if first_number < 10:
            return f"Низкая энергия (число {first_number}) - рекомендуется отдых и восстановление"
        elif first_number < 20:
            return f"Сбалансированная энергия (число {first_number}) - стабильность в действиях"
        else:
            return f"Высокая энергия (число {first_number}) - время активных действий"

    def _analyze_life_purpose(self, fourth_number):
        """Анализ жизненного предназначения - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        # Основано на нумерологическом значении четвертого числа
        purposes = {
            1: f"Лидерство и инициатива (число {fourth_number}) - ваше призвание вести за собой",
            2: f"Гармония и сотрудничество (число {fourth_number}) - ваш путь в партнерстве",
            3: f"Творчество и самовыражение (число {fourth_number}) - ваша миссия в искусстве",
            4: f"Стабильность и порядок (число {fourth_number}) - ваша задача в организации",
            5: f"Свобода и изменения (число {fourth_number}) - ваша судьба в трансформациях",
            6: f"Семья и ответственность (число {fourth_number}) - ваше предназначение в заботе",
            7: f"Знания и анализ (число {fourth_number}) - ваш дар в исследованиях",
            8: f"Деньги и власть (число {fourth_number}) - ваша сила в управлении",
            9: f"Служение и гуманизм (число {fourth_number}) - ваше призвание в помощи людям"
        }
        return purposes.get(fourth_number,
                            f"Многогранное предназначение (число {fourth_number}) - исследуйте разные пути")

    def _analyze_talents(self, matrix):
        """Анализ талантов на основе матрицы - ТОЛЬКО НА ОСНОВЕ РАСЧЕТОВ"""
        talents = []

        # Каждый талант основан на конкретных расчетах матрицы
        if matrix.get('3', 0) >= 2:
            talents.append(f"Технические способности (цифра 3: {matrix.get('3', 0)})")
        if matrix.get('5', 0) >= 1:
            talents.append(f"Интуиция и предвидение (цифра 5: {matrix.get('5', 0)})")
        if matrix.get('7', 0) >= 2:
            talents.append(f"Творческие способности (цифра 7: {matrix.get('7', 0)})")
        if matrix.get('9', 0) >= 2:
            talents.append(f"Аналитический ум (цифра 9: {matrix.get('9', 0)})")
        if matrix.get('2', 0) >= 2:
            talents.append(f"Экстрасенсорные способности (цифра 2: {matrix.get('2', 0)})")

        # Если талантов не найдено по расчетам, анализируем доминирующие цифры
        if not talents:
            max_digit = max(matrix.items(), key=lambda x: x[1])
            if max_digit[1] > 0:
                talents.append(f"Практические навыки (доминирующая цифра {max_digit[0]}: {max_digit[1]})")

        return talents
backend.user_services.py:
from backend.database import async_session, User
from sqlalchemy.future import select
import logging

logger = logging.getLogger(__name__)


async def create_or_update_user(
        telegram_id: int,
        birth_date,
        birth_time,
        birth_city: str,
        profession: str = None,
        job_position: str = None,
        current_city: str = None
):
    """Создание или обновление пользователя"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()

            if user:
                # Обновляем существующего пользователя
                user.birth_date = birth_date
                user.birth_time = birth_time
                user.birth_city = birth_city
                if profession: user.profession = profession
                if job_position: user.job_position = job_position
                if current_city: user.current_city = current_city
                logger.info(f"📝 Обновлен пользователь {telegram_id}")
            else:
                # Создаем нового пользователя
                user = User(
                    telegram_id=telegram_id,
                    birth_date=birth_date,
                    birth_time=birth_time,
                    birth_city=birth_city,
                    profession=profession,
                    job_position=job_position,
                    current_city=current_city
                )
                session.add(user)
                logger.info(f"🆕 Создан новый пользователь {telegram_id}")

            await session.commit()
            return user

    except Exception as e:
        logger.error(f"❌ Ошибка при работе с пользователем {telegram_id}: {e}")
        raise


async def get_user_profile(telegram_id: int):
    """Получение профиля пользователя"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()

            if user:
                return {
                    'telegram_id': user.telegram_id,
                    'birth_date': user.birth_date,
                    'birth_time': user.birth_time,
                    'birth_city': user.birth_city,
                    'profession': user.profession,
                    'job_position': user.job_position,
                    'current_city': user.current_city,
                    'created_at': user.created_at
                }
            return None

    except Exception as e:
        logger.error(f"❌ Ошибка при получении профиля {telegram_id}: {e}")
        return None


async def update_user_profession(telegram_id: int, profession: str, job_position: str = None):
    """Обновление профессиональных данных пользователя"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()

            if user:
                user.profession = profession
                if job_position:
                    user.job_position = job_position
                await session.commit()
                logger.info(f"📝 Обновлены профессиональные данные для {telegram_id}")
                return user
            else:
                raise ValueError("Пользователь не найден")

    except Exception as e:
        logger.error(f"❌ Ошибка при обновлении профессии {telegram_id}: {e}")
        raise

bot:
bot.config.py:
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
bot.handlers.py:
from aiogram import Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from datetime import datetime, date
import logging

from backend.assistant import assistant

logger = logging.getLogger(__name__)

# Создаем роутер
router = Router()


# Определяем состояния для сбора данных
class DataCollectionStates(StatesGroup):
    waiting_for_birth_date = State()
    waiting_for_birth_time = State()
    waiting_for_birth_city = State()
    waiting_for_current_city = State()
    waiting_for_profession = State()
    waiting_for_job_position = State()


# Создаем клавиатуру с двумя основными кнопками
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📊 Расчет натальной карты")],
            [KeyboardButton(text="📅 Рекомендации на сегодня")],
        ],
        resize_keyboard=True
    )


@router.message(Command("start"))
async def cmd_start(message: types.Message):
    """Команда начала работы с ботом"""
    welcome_text = """
👋 Добро пожаловать в ваш персональный ассистент!

Я помогу вам получать персонализированные рекомендации на основе:
• 🌟 Натальной карты и астрологических транзитов
• 🔢 Психоматрицы по дате рождения  
• ⚡ Биоритмов на каждый день
• 💼 Вашей профессиональной деятельности

Выберите действие из меню ниже:
    """

    await message.answer(welcome_text, reply_markup=get_main_keyboard())


@router.message(lambda message: message.text == "📊 Расчет натальной карты")
async def start_data_collection(message: types.Message, state: FSMContext):
    """Начало сбора данных пользователя"""

    # Проверяем статус данных пользователя
    status = await assistant.get_user_data_status(message.from_user.id)

    if status['is_complete']:
        await message.answer(
            "✅ Ваши основные данные уже собраны!\n"
            "Если хотите обновить профессию или город, используйте соответствующую команду.",
            reply_markup=get_main_keyboard()
        )
    else:
        await message.answer(
            "📊 Начнем сбор данных для персонализированных рекомендаций!\n\n"
            "Пожалуйста, введите вашу дату рождения в формате ГГГГ-ММ-ДД:",
            reply_markup=types.ReplyKeyboardRemove()
        )
        await state.set_state(DataCollectionStates.waiting_for_birth_date)


@router.message(DataCollectionStates.waiting_for_birth_date)
async def process_birth_date(message: types.Message, state: FSMContext):
    """Обработка даты рождения"""
    try:
        birth_date = datetime.strptime(message.text, "%Y-%m-%d").date()
        await state.update_data(birth_date=birth_date)

        await message.answer(
            "✅ Дата рождения сохранена!\n\n"
            "Теперь введите время рождения в формате ЧЧ:ММ (24 часа):"
        )
        await state.set_state(DataCollectionStates.waiting_for_birth_time)

    except ValueError:
        await message.answer("❌ Неверный формат даты. Используйте формат ГГГГ-ММ-ДД:")


@router.message(DataCollectionStates.waiting_for_birth_time)
async def process_birth_time(message: types.Message, state: FSMContext):
    """Обработка времени рождения"""
    try:
        birth_time = datetime.strptime(message.text, "%H:%M").time()
        await state.update_data(birth_time=birth_time)

        await message.answer(
            "✅ Время рождения сохранено!\n\n"
            "Введите город рождения:"
        )
        await state.set_state(DataCollectionStates.waiting_for_birth_city)

    except ValueError:
        await message.answer("❌ Неверный формат времени. Используйте формат ЧЧ:ММ:")


@router.message(DataCollectionStates.waiting_for_birth_city)
async def process_birth_city(message: types.Message, state: FSMContext):
    """Обработка города рождения"""
    birth_city = message.text.strip()
    await state.update_data(birth_city=birth_city)

    await message.answer(
        "✅ Город рождения сохранен!\n\n"
        "Теперь введите город проживания:"
    )
    await state.set_state(DataCollectionStates.waiting_for_current_city)


@router.message(DataCollectionStates.waiting_for_current_city)
async def process_current_city(message: types.Message, state: FSMContext):
    """Обработка города проживания"""
    current_city = message.text.strip()
    await state.update_data(current_city=current_city)

    await message.answer(
        "✅ Город проживания сохранен!\n\n"
        "Введите вашу специальность или профессию:"
    )
    await state.set_state(DataCollectionStates.waiting_for_profession)


@router.message(DataCollectionStates.waiting_for_profession)
async def process_profession(message: types.Message, state: FSMContext):
    """Обработка профессии"""
    profession = message.text.strip()
    await state.update_data(profession=profession)

    await message.answer(
        "✅ Профессия сохранена!\n\n"
        "Введите вашу должность (если нет - напишите 'нет'):"
    )
    await state.set_state(DataCollectionStates.waiting_for_job_position)


@router.message(DataCollectionStates.waiting_for_job_position)
async def process_job_position(message: types.Message, state: FSMContext):
    """Обработка должности и завершение сбора данных"""
    job_position = message.text.strip()
    if job_position.lower() == 'нет':
        job_position = None

    user_data = await state.get_data()

    try:
        # Сохраняем все данные через ассистента
        result = await assistant.collect_user_data(
            telegram_id=message.from_user.id,
            birth_date=user_data['birth_date'],
            birth_time=user_data['birth_time'],
            birth_city=user_data['birth_city'],
            current_city=user_data['current_city'],
            profession=user_data['profession'],
            job_position=job_position
        )

        if result['success']:
            await message.answer(
                "🎉 Поздравляем! Все данные успешно собраны!\n\n"
                "Теперь вы можете получать персонализированные рекомендации:",
                reply_markup=get_main_keyboard()
            )
        else:
            await message.answer(
                f"❌ {result['message']}\n\n"
                "Попробуйте начать сбор данных заново.",
                reply_markup=get_main_keyboard()
            )

    except Exception as e:
        await message.answer(
            f"❌ Произошла ошибка при сохранении данных: {str(e)}\n\n"
            "Попробуйте начать сбор данных заново.",
            reply_markup=get_main_keyboard()
        )

    await state.clear()


@router.message(lambda message: message.text == "📅 Рекомендации на сегодня")
async def get_todays_recommendations(message: types.Message):
    """Получение рекомендаций на сегодня"""

    # Проверяем наличие данных
    status = await assistant.get_user_data_status(message.from_user.id)
    if not status['is_complete']:
        await message.answer(
            "❌ Сначала необходимо собрать данные для рекомендаций!\n"
            "Нажмите '📊 Расчет натальной карты'",
            reply_markup=get_main_keyboard()
        )
        return

    processing_msg = await message.answer("🔄 Формирую рекомендации на сегодня...")

    try:
        result = await assistant.get_todays_recommendations(message.from_user.id)

        if result['success']:
            await message.answer(result['recommendations'], parse_mode="Markdown")
        else:
            await message.answer(result['message'])

    except Exception as e:
        logger.error(f"Ошибка получения рекомендаций на сегодня: {e}")
        await message.answer(
            "❌ Произошла ошибка при формировании рекомендаций\n"
            "Попробуйте позже или обратитесь в поддержку."
        )

    await processing_msg.delete()


@router.message()
async def handle_other_messages(message: types.Message):
    """Обработка всех остальных сообщений"""
    await message.answer(
        "Выберите действие из меню ниже:",
        reply_markup=get_main_keyboard()
    )
bot.main.py:
from aiogram import Bot, Dispatcher
import asyncio
import logging

from bot.config import TOKEN
from bot.handlers import router
from backend.db_connection import check_db_connection
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BiorhythmCalculator:
    """
    Калькулятор биоритмов на основе даты рождения.
    Рассчитывает физический, эмоциональный и интеллектуальный циклы.
    """

    def __init__(self):
        # Периоды биоритмов в днях
        self.PHYSICAL_CYCLE = 23
        self.EMOTIONAL_CYCLE = 28
        self.INTELLECTUAL_CYCLE = 33
        self.INTUITIVE_CYCLE = 38  # Дополнительный цикл

    def calculate_biorhythms(self, birth_date: date, target_date: date) -> Dict:
        """
        Расчет биоритмов на заданную дату

        Args:
            birth_date: Дата рождения
            target_date: Дата для расчета

        Returns:
            Словарь с данными биоритмов
        """
        try:
            # Вычисляем количество прожитых дней
            days_lived = (target_date - birth_date).days

            if days_lived < 0:
                raise ValueError("Дата расчета не может быть раньше даты рождения")

            # Рассчитываем фазы биоритмов
            physical = self._calculate_cycle(days_lived, self.PHYSICAL_CYCLE)
            emotional = self._calculate_cycle(days_lived, self.EMOTIONAL_CYCLE)
            intellectual = self._calculate_cycle(days_lived, self.INTELLECTUAL_CYCLE)
            intuitive = self._calculate_cycle(days_lived, self.INTUITIVE_CYCLE)

            # Общий показатель энергии
            overall_energy = self._calculate_overall_energy(physical, emotional, intellectual, intuitive)

            # Рекомендации на основе биоритмов
            recommendations = self._generate_recommendations(physical, emotional, intellectual, intuitive,
                                                             overall_energy)

            biorhythm_data = {
                'calculation_date': target_date.isoformat(),
                'days_lived': days_lived,
                'cycles': {
                    'physical': physical,
                    'emotional': emotional,
                    'intellectual': intellectual,
                    'intuitive': intuitive
                },
                'overall_energy': overall_energy,
                'recommendations': recommendations,
                'critical_days': self._find_critical_days(physical, emotional, intellectual, target_date),
                'peak_days': self._find_peak_days(physical, emotional, intellectual, target_date)
            }

            logger.info(f"✅ Биоритмы рассчитаны для {target_date}, прожито дней: {days_lived}")
            return biorhythm_data

        except Exception as e:
            logger.error(f"❌ Ошибка расчета биоритмов: {e}")
            raise

    def _calculate_cycle(self, days_lived: int, cycle_length: int) -> Dict:
        """
        Расчет одного цикла биоритма

        Args:
            days_lived: Количество прожитых дней
            cycle_length: Длина цикла в днях

        Returns:
            Данные цикла
        """
        # Текущая фаза в радианах (2π за полный цикл)
        phase = (2 * math.pi * days_lived) / cycle_length

        # Значение синусоиды (-1 до +1)
        value = math.sin(phase)

        # Процент от максимума (0% до 100%)
        percentage = ((value + 1) / 2) * 100

        # День в цикле (0 до cycle_length-1)
        day_in_cycle = days_lived % cycle_length

        return {
            'value': round(value, 4),
            'percentage': round(percentage, 2),
            'day_in_cycle': day_in_cycle,
            'phase': self._get_phase_description(value),
            'trend': self._get_trend(phase)
        }

    def _get_phase_description(self, value: float) -> str:
        """Описание фазы биоритма"""
        if value >= 0.7:
            return "пик энергии"
        elif value >= 0.3:
            return "высокая активность"
        elif value >= -0.3:
            return "нейтральная фаза"
        elif value >= -0.7:
            return "низкая активность"
        else:
            return "критическая точка"

    def _get_trend(self, phase: float) -> str:
        """Определение тренда (растет/падает)"""
        # Анализируем производную (cos(phase))
        derivative = math.cos(phase)

        if derivative > 0.1:
            return "растет"
        elif derivative < -0.1:
            return "падает"
        else:
            return "стабильно"

    def _calculate_overall_energy(self, physical: Dict, emotional: Dict, intellectual: Dict, intuitive: Dict) -> Dict:
        """Расчет общего уровня энергии"""
        # Взвешенная сумма всех циклов
        total_energy = (
                physical['value'] * 0.3 +  # Физический цикл - 30%
                emotional['value'] * 0.25 +  # Эмоциональный - 25%
                intellectual['value'] * 0.25 +  # Интеллектуальный - 25%
                intuitive['value'] * 0.2  # Интуитивный - 20%
        )

        # Нормализуем до 0-100%
        energy_percentage = ((total_energy + 1) / 2) * 100

        # Определяем уровень энергии
        if energy_percentage >= 80:
            level = "очень высокий"
            description = "Отличный день для активных действий и важных решений"
        elif energy_percentage >= 60:
            level = "высокий"
            description = "Хороший день для продуктивной работы"
        elif energy_percentage >= 40:
            level = "средний"
            description = "Стабильный день, подходит для рутинных задач"
        elif energy_percentage >= 20:
            level = "низкий"
            description = "День для отдыха и восстановления сил"
        else:
            level = "очень низкий"
            description = "Рекомендуется беречь энергию, избегать нагрузок"

        return {
            'value': round(total_energy, 4),
            'percentage': round(energy_percentage, 2),
            'level': level,
            'description': description
        }

    def _generate_recommendations(self, physical: Dict, emotional: Dict, intellectual: Dict, intuitive: Dict,
                                  overall: Dict) -> List[str]:
        """Генерация рекомендаций на основе биоритмов"""
        recommendations = []

        # Физические рекомендации
        if physical['value'] > 0.5:
            recommendations.append("💪 Идеальный день для спорта и физической активности")
        elif physical['value'] < -0.5:
            recommendations.append("🛌 Избегайте тяжелых физических нагрузок")

        # Эмоциональные рекомендации
        if emotional['value'] > 0.6:
            recommendations.append("😊 Отличное время для общения и новых знакомств")
        elif emotional['value'] < -0.4:
            recommendations.append("🧘 Контролируйте эмоции, избегайте конфликтов")

        # Интеллектуальные рекомендации
        if intellectual['value'] > 0.5:
            recommendations.append("📚 Благоприятный период для обучения и анализа")
        elif intellectual['value'] < -0.3:
            recommendations.append("📝 Отложите сложные интеллектуальные задачи")

        # Интуитивные рекомендации
        if intuitive['value'] > 0.4:
            recommendations.append("🔮 Доверяйте интуиции при принятии решений")

        # Общие рекомендации по энергии
        if overall['percentage'] > 70:
            recommendations.append("🚀 Используйте высокую энергию для важных проектов")
        elif overall['percentage'] < 30:
            recommendations.append("⚡ Экономьте силы, планируйте короткие перерывы")

        # Если рекомендаций мало, добавляем общие
        if len(recommendations) < 3:
            recommendations.extend([
                "📅 Следуйте своему естественному ритму",
                "⏰ Планируйте задачи в соответствии с энергетическими пиками",
                "💧 Пейте足够 воды для поддержания энергии"
            ])

        return recommendations[:5]  # Не более 5 рекомендаций

    def _find_critical_days(self, physical: Dict, emotional: Dict, intellectual: Dict, target_date: date) -> List[Dict]:
        """Определение критических дней (ближайшие 7 дней)"""
        critical_days = []

        # Проверяем текущий день
        if (abs(physical['value']) > 0.9 or
                abs(emotional['value']) > 0.9 or
                abs(intellectual['value']) > 0.9):
            critical_days.append({
                'date': target_date.isoformat(),
                'cycles': self._get_critical_cycles(physical, emotional, intellectual),
                'description': 'Критический день - будьте осторожны'
            })

        return critical_days

    def _find_peak_days(self, physical: Dict, emotional: Dict, intellectual: Dict, target_date: date) -> List[Dict]:
        """Определение пиковых дней (ближайшие 7 дней)"""
        peak_days = []

        # Проверяем текущий день
        if (physical['value'] > 0.8 or
                emotional['value'] > 0.8 or
                intellectual['value'] > 0.8):

            peak_cycles = []
            if physical['value'] > 0.8: peak_cycles.append('физический')
            if emotional['value'] > 0.8: peak_cycles.append('эмоциональный')
            if intellectual['value'] > 0.8: peak_cycles.append('интеллектуальный')

            peak_days.append({
                'date': target_date.isoformat(),
                'cycles': peak_cycles,
                'description': f'Пик энергии в циклах: {", ".join(peak_cycles)}'
            })

        return peak_days

    def _get_critical_cycles(self, physical: Dict, emotional: Dict, intellectual: Dict) -> List[str]:
        """Получение списка критических циклов"""
        critical = []
        if abs(physical['value']) > 0.9: critical.append('физический')
        if abs(emotional['value']) > 0.9: critical.append('эмоциональный')
        if abs(intellectual['value']) > 0.9: critical.append('интеллектуальный')
        return critical

    def calculate_weekly_forecast(self, birth_date: date, start_date: date, days: int = 7) -> List[Dict]:
        """Расчет прогноза биоритмов на несколько дней"""
        forecast = []

        for i in range(days):
            current_date = start_date + timedelta(days=i)
            biorhythms = self.calculate_biorhythms(birth_date, current_date)

            forecast.append({
                'date': current_date.isoformat(),
                'overall_energy': biorhythms['overall_energy']['percentage'],
                'physical': biorhythms['cycles']['physical']['percentage'],
                'emotional': biorhythms['cycles']['emotional']['percentage'],
                'intellectual': biorhythms['cycles']['intellectual']['percentage'],
                'is_critical': len(biorhythms['critical_days']) > 0,
                'is_peak': len(biorhythms['peak_days']) > 0
            })

        return forecast
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def main():
    try:
        # Проверяем подключение к БД перед запуском
        logger.info("🔍 Проверка подключения к базе данных...")
        db_connected = await check_db_connection()

        if not db_connected:
            logger.error("❌ Не удалось подключиться к базе данных. Завершение работы.")
            return

        bot = Bot(token=TOKEN)
        dp = Dispatcher()

        # Подключаем роутер
        dp.include_router(router)

        logger.info("✅ Бот запущен и готов к работе...")
        logger.info("✅ База данных подключена успешно")
        logger.info("✅ Personal Assistant инициализирован")

        # Запускаем поллинг
        await dp.start_polling(bot)

    except Exception as e:
        logger.error(f"❌ Ошибка при запуске бота: {e}")
    finally:
        if 'bot' in locals():
            await bot.close()
        logger.info("🛑 Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())



docker-compose.yml:
version: '3.8'

services:
  # Основная база данных для проекта Astra
  postgres:
    image: postgres:16
    container_name: postgres_astra
    environment:
      POSTGRES_DB: p_assistant_bd
      POSTGRES_USER: pers_assist
      POSTGRES_PASSWORD: astra123
    ports:
      - "5435:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts/01-init-tables.sql:/docker-entrypoint-initdb.d/01-init-tables.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pers_assist -d p_assistant_bd"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Бот для Telegram - основной интерфейс пользователя
  astra_bot:
    build: .
    container_name: astra_bot
    environment:
      DATABASE_URL: "postgresql+asyncpg://pers_assist:astra123@postgres:5432/p_assistant_bd"
      TELEGRAM_BOT_TOKEN: "${TELEGRAM_BOT_TOKEN}"
      LOG_LEVEL: "INFO"
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped
    command: python bot/main.py
    volumes:
      - .:/app
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # API для доступа к данным Astra - для интеграции с другими проектами
  astra_api:
    build: .
    container_name: astra_api
    environment:
      DATABASE_URL: "postgresql+asyncpg://pers_assist:astra123@postgres:5432/p_assistant_bd"
      TELEGRAM_BOT_TOKEN: "${TELEGRAM_BOT_TOKEN}"
      ASSISTANT_API_HOST: "0.0.0.0"
      ASSISTANT_API_PORT: "8000"
      LOG_LEVEL: "INFO"
    ports:
      - "8000:8000"  # API порт для внешних проектов (assistant и другие)
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped
    command: python -m backend.assistant_api
    volumes:
      - .:/app
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:

networks:
  default:
    name: astra_network
    
Dockerfile:
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директории для логов
RUN mkdir -p /app/logs

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Команда по умолчанию (будет переопределена в docker-compose)
CMD ["python", "bot/main.py"]

init-scripts.01-init-tables.sql:
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


requirements.txt
.env
.gitignore


