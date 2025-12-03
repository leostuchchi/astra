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