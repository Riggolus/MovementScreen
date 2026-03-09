"""App-wide settings loaded from environment / .env file."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/movementscreen"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24        # 24 h
    refresh_token_expire_days: int = 30

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
