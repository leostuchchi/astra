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