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
