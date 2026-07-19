import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    jwt_secret_key: str = "insecure-dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 120

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vlm_model: str = "Salesforce/blip-image-captioning-base"
    device: str = "cpu"

    data_dir: str = "./data"
    sqlite_path: str = "./data/app.db"

    default_admin_username: str = "admin"
    default_admin_password: str = "change_this_password"

    # When false (set this on the deployed/Render instance), upload and
    # delete endpoints are locked. Process/upload books locally where the
    # embedding + VLM models run, then deploy the populated data/ folder.
    enable_upload_delete: bool = True

    # Comma-separated list of allowed frontend origins for CORS.
    # e.g. "http://localhost:4200,https://your-frontend.onrender.com"
    frontend_origins: str = "http://localhost:4200"

    class Config:
        env_file = ".env"


settings = Settings()
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
Path(settings.data_dir, "pdfs").mkdir(parents=True, exist_ok=True)
Path(settings.data_dir, "index").mkdir(parents=True, exist_ok=True)
