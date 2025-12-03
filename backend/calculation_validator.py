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