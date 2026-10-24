"""Configuration du service, lue depuis les variables d'environnement (12-factor).

Exemple : MEDXRAY_MODEL_PATH=/app/models/model.pt MEDXRAY_MAX_UPLOAD_MB=5
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDXRAY_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    model_path: Path = Path("models/model.pt")
    max_upload_mb: float = Field(default=10.0, gt=0, le=50)
    # 0 = laisser PyTorch décider. En conteneur, fixer au nombre de CPU alloués évite la
    # sur-souscription (PyTorch voit les CPU de l'hôte, pas le quota du conteneur).
    torch_threads: int = Field(default=0, ge=0)
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=list)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
