"""
Central application configuration.

Reads settings from environment variables / a local .env file so that no
secrets or absolute paths are hardcoded anywhere else in the codebase.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Emergency Message Prioritization System"
    app_env: str = "development"
    debug: bool = True

    database_url: str = "sqlite:///./data/emergency.db"

    secret_key: str = "dev-only-not-secure-change-me"
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"

    nlp_model_dir: str = "./nlp/artifacts"
    rl_model_dir: str = "./rl/artifacts"

    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    seed_demo_data: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def nlp_model_path(self) -> Path:
        return (BASE_DIR / self.nlp_model_dir).resolve()

    @property
    def rl_model_path(self) -> Path:
        return (BASE_DIR / self.rl_model_dir).resolve()


settings = Settings()
