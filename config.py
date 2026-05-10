"""Configuration loader with Azure Key Vault + env var fallback."""
import os
import logging
from functools import lru_cache
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


"""Configuration loader with Azure Key Vault + env var fallback."""
# ... (existing imports unchanged)


class Settings(BaseSettings):
    """Application settings loaded from Azure Key Vault or env vars."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # ===== Key Vault =====
    azure_key_vault_url: Optional[str] = None
    use_key_vault: bool = False
    
    # ===== App =====
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    
    # ===== Caching =====
    cache_backend: str = Field(
        default="memory",
        description="Cache backend: 'memory' or 'redis'",
    )
    redis_url: str = Field(
        default="",
        description="Redis URL, e.g., redis://localhost:6379/0",
    )
    cache_max_size: int = Field(
        default=1000,
        description="Max items in in-memory cache",
    )
    
    # Per-tool TTLs (seconds)
    cache_ttl_geocoding: int = Field(default=86400, description="24 hours")
    cache_ttl_current_weather: int = Field(default=600, description="10 minutes")
    cache_ttl_forecast: int = Field(default=1800, description="30 minutes")
    cache_ttl_agent_response: int = Field(default=300, description="5 minutes")
    
    # Enable/disable cache layers
    enable_tool_cache: bool = True
    enable_response_cache: bool = True

class KeyVaultLoader:
    """Lazy loader for Azure Key Vault secrets with caching."""
    
    # Map setting names → Key Vault secret names
    # KV doesn't allow underscores in secret names, only hyphens
    SECRET_MAPPING = {
        "openai_api_key": "openai-api-key",
        "openai_model": "openai-model",
    }
    
    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        self._client: Optional[SecretClient] = None
        self._cache: dict[str, str] = {}
    
    @property
    def client(self) -> SecretClient:
        """Lazy-init the Key Vault client."""
        if self._client is None:
            logger.info(f"Initializing Key Vault client for {self.vault_url}")
            credential = DefaultAzureCredential(
                # Tune timeouts/retries for production
                exclude_interactive_browser_credential=True,
            )
            self._client = SecretClient(
                vault_url=self.vault_url,
                credential=credential,
            )
        return self._client
    
    def get_secret(self, setting_name: str, default: str = "") -> str:
        """Fetch a secret by its setting name (uses SECRET_MAPPING)."""
        if setting_name in self._cache:
            return self._cache[setting_name]
        
        secret_name = self.SECRET_MAPPING.get(setting_name)
        if not secret_name:
            return default
        
        try:
            secret = self.client.get_secret(secret_name)
            value = secret.value or default
            self._cache[setting_name] = value
            logger.info(f"Loaded secret '{secret_name}' from Key Vault")
            return value
        except ResourceNotFoundError:
            logger.warning(f"Secret '{secret_name}' not found in Key Vault")
            return default
        except ClientAuthenticationError as e:
            logger.error(f"Auth failed for Key Vault: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load secret '{secret_name}': {e}")
            return default
    
    def refresh(self) -> None:
        """Clear the cache to force re-fetching secrets."""
        self._cache.clear()
        logger.info("Key Vault cache cleared")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build Settings, optionally pulling from Azure Key Vault.
    
    Precedence:
      1. If USE_KEY_VAULT=true and AZURE_KEY_VAULT_URL is set → load from KV
      2. Otherwise → load from env vars / .env
    """
    settings = Settings()
    
    if settings.use_key_vault and settings.azure_key_vault_url:
        logger.info("Loading configuration from Azure Key Vault")
        loader = KeyVaultLoader(settings.azure_key_vault_url)
        
        # Override settings with Key Vault values where present
        for setting_name in KeyVaultLoader.SECRET_MAPPING:
            kv_value = loader.get_secret(setting_name, default="")
            if kv_value:
                setattr(settings, setting_name, kv_value)
    else:
        logger.info("Loading configuration from environment variables")
    
    # Validate required settings
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required (set via env var or Key Vault secret 'openai-api-key')"
        )
    
    return settings
