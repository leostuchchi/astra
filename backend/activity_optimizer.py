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