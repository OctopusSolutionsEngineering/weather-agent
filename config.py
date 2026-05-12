"""Configuration loader using Azure App Configuration + Key Vault."""
import os
import logging
from functools import lru_cache
from threading import Lock
from typing import Optional, Any

from azure.identity import DefaultAzureCredential
from azure.appconfiguration.provider import (
    load,
    SettingSelector,
    WatchKey,
)
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================
# Bootstrap settings (always from env — needed to find Azure)
# ============================================================

class BootstrapSettings(BaseSettings):
    """Minimal config needed to BOOTSTRAP from Azure.
    
    These MUST come from environment / K8s ConfigMap because we
    need them before we can reach Azure App Configuration.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # App Configuration
    azure_appconfig_endpoint: Optional[str] = Field(
        default=None,
        description="e.g., https://my-appconfig.azconfig.io",
    )
    use_app_configuration: bool = False
    appconfig_label: str = Field(
        default="dev",
        description="App Config label (e.g., 'dev', 'staging', 'prod')",
    )
    appconfig_refresh_interval: int = Field(
        default=30,
        description="Seconds between refresh checks",
    )
    
    # Key Vault (fallback when not using App Config)
    azure_key_vault_url: Optional[str] = None
    use_key_vault: bool = False


# ============================================================
# Application settings (loaded from App Config or env)
# ============================================================

class Settings(BaseSettings):
    """All application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # ===== Bootstrap (carried through for reference) =====
    azure_appconfig_endpoint: Optional[str] = None
    use_app_configuration: bool = False
    appconfig_label: str = "prod"
    azure_key_vault_url: Optional[str] = None
    use_key_vault: bool = False
    
    # ===== Secrets (from Key Vault) =====
    openai_api_key: str = ""
    
    # ===== App config (from App Configuration) =====
    openai_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    
    # ===== Feature flags =====
    feature_response_cache: bool = True
    feature_tool_cache: bool = True
    feature_streaming: bool = False
    feature_strict_mode: bool = False  # e.g., enforce stricter validation
    
    # ===== Caching =====
    cache_backend: str = "memory"
    redis_url: str = ""
    cache_max_size: int = 1000
    cache_ttl_geocoding: int = 86400
    cache_ttl_current_weather: int = 600
    cache_ttl_forecast: int = 1800
    cache_ttl_agent_response: int = 300


# ============================================================
# App Configuration Loader
# ============================================================

class AppConfigLoader:
    """Loads config from Azure App Configuration with KV reference resolution."""
    
    # Map App Config keys → Settings attribute names
    # Use App Config "key prefix" pattern for organization:
    #   weather-agent:openai:model
    #   weather-agent:cache:backend
    #   weather-agent:feature:streaming
    KEY_PREFIX = "weather-agent:"
    
    # Feature flags use a separate namespace in App Config
    FEATURE_FLAG_PREFIX = ".appconfig.featureflag/weather-agent-"
    
    def __init__(
        self,
        endpoint: str,
        label: str = "prod",
        refresh_interval: int = 30,
    ):
        self.endpoint = endpoint
        self.label = label
        self.refresh_interval = refresh_interval
        self._provider = None
        self._lock = Lock()
        
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=True,
        )
        self._credential = credential
    
    def _build_provider(self):
        """Initialize the App Configuration provider with auto-refresh."""
        logger.info(
            f"Loading App Configuration from {self.endpoint} (label={self.label})"
        )
        
        # SettingSelector picks which keys to load
        selector = SettingSelector(
            key_filter=f"{self.KEY_PREFIX}*",
            label_filter=self.label,
        )
        
        # WatchKey enables auto-refresh: when this key changes,
        # the entire config bundle is re-fetched
        sentinel = WatchKey(
            key=f"{self.KEY_PREFIX}sentinel",
            label=self.label,
        )
        
        return load(
            endpoint=self.endpoint,
            credential=self._credential,
            selects=[selector],
            refresh_on=[sentinel],
            refresh_interval=self.refresh_interval,
            feature_flag_enabled=True,
            feature_flag_refresh_enabled=True,
            # Trim the prefix from keys for cleaner access:
            # weather-agent:openai:model → openai:model
            trim_prefixes=[self.KEY_PREFIX],
            # Key Vault references resolved automatically with same credential
            keyvault_credential=self._credential,
        )
    
    @property
    def provider(self):
        """Lazy-init the provider."""
        if self._provider is None:
            with self._lock:
                if self._provider is None:
                    self._provider = self._build_provider()
        return self._provider
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from App Config (with KV references resolved)."""
        return self.provider.get(key, default)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check a feature flag."""
        # The provider exposes feature flags via 'feature_management'
        features = self.provider.get("feature_management", {}).get("feature_flags", [])
        for f in features:
            if f.get("id") == feature_name:
                return f.get("enabled", False)
        return False
    
    def refresh(self) -> None:
        """Manually trigger a config refresh."""
        self.provider.refresh()
        logger.info("App Configuration refreshed")
    
    def all_keys(self) -> list[str]:
        """Return all loaded keys (for debugging)."""
        return list(self.provider.keys())


# ============================================================
# Standalone Key Vault Loader (used when App Config is off)
# ============================================================

class KeyVaultLoader:
    """Fallback secret loader when App Configuration is disabled."""
    
    SECRET_MAPPING = {"openai_api_key": "openai-api-key"}
    
    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        self._client: Optional[SecretClient] = None
        self._cache: dict[str, str] = {}
    
    @property
    def client(self) -> SecretClient:
        if self._client is None:
            credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=True,
            )
            self._client = SecretClient(vault_url=self.vault_url, credential=credential)
        return self._client
    
    def get_secret(self, setting_name: str, default: str = "") -> str:
        if setting_name in self._cache:
            return self._cache[setting_name]
        secret_name = self.SECRET_MAPPING.get(setting_name)
        if not secret_name:
            return default
        try:
            value = self.client.get_secret(secret_name).value or default
            self._cache[setting_name] = value
            return value
        except ResourceNotFoundError:
            return default


# ============================================================
# Main entry point
# ============================================================

# App Config keys → Settings attributes
# Keys are stored in App Config WITHOUT the "weather-agent:" prefix
# (because the loader trims it).
APPCONFIG_KEY_MAP = {
    # OpenAI
    "openai:model": ("openai_model", str),
    "openai:api-key": ("openai_api_key", str),  # KV reference!
    
    # General
    "log-level": ("log_level", str),
    
    # Cache
    "cache:backend": ("cache_backend", str),
    "cache:redis-url": ("redis_url", str),
    "cache:max-size": ("cache_max_size", int),
    "cache:ttl:geocoding": ("cache_ttl_geocoding", int),
    "cache:ttl:current-weather": ("cache_ttl_current_weather", int),
    "cache:ttl:forecast": ("cache_ttl_forecast", int),
    "cache:ttl:agent-response": ("cache_ttl_agent_response", int),
}

# Feature flag names in App Config → Settings attributes
FEATURE_FLAG_MAP = {
    "response-cache": "feature_response_cache",
    "tool-cache": "feature_tool_cache",
    "streaming": "feature_streaming",
    "strict-mode": "feature_strict_mode",
}


_app_config_loader: Optional[AppConfigLoader] = None
_settings: Optional[Settings] = None
_settings_lock = Lock()


def _coerce(value: Any, target_type: type) -> Any:
    """Coerce string values from App Config to proper Python types."""
    if value is None:
        return None
    if target_type is bool:
        return str(value).lower() in {"true", "1", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _populate_from_app_config(settings: Settings, loader: AppConfigLoader) -> None:
    """Apply App Config values onto the Settings object."""
    for ac_key, (attr_name, type_) in APPCONFIG_KEY_MAP.items():
        value = loader.get(ac_key)
        if value is not None:
            setattr(settings, attr_name, _coerce(value, type_))
            logger.debug(f"App Config: {ac_key} → settings.{attr_name}")
    
    for flag_name, attr_name in FEATURE_FLAG_MAP.items():
        # Feature flag IDs in App Config typically look like:
        #   weather-agent-streaming, weather-agent-response-cache
        feature_id = f"weather-agent-{flag_name}"
        enabled = loader.is_feature_enabled(feature_id)
        setattr(settings, attr_name, enabled)
        logger.debug(f"Feature flag {feature_id} = {enabled}")


def get_app_config_loader() -> Optional[AppConfigLoader]:
    """Return the singleton App Config loader, if enabled."""
    global _app_config_loader
    if _app_config_loader is not None:
        return _app_config_loader
    
    bootstrap = BootstrapSettings()
    if bootstrap.use_app_configuration and bootstrap.azure_appconfig_endpoint:
        _app_config_loader = AppConfigLoader(
            endpoint=bootstrap.azure_appconfig_endpoint,
            label=bootstrap.appconfig_label,
            refresh_interval=bootstrap.appconfig_refresh_interval,
        )
    return _app_config_loader


def get_settings() -> Settings:
    """Get current settings.
    
    Important: This is NOT cached with @lru_cache anymore, because
    we want config to reflect live App Configuration updates.
    
    However, the underlying App Config provider has its own refresh
    mechanism with `refresh_interval`, so we're not hammering Azure.
    """
    global _settings
    
    with _settings_lock:
        # First time: build the settings object
        if _settings is None:
            _settings = Settings()
            bootstrap = BootstrapSettings()
            
            # Copy bootstrap fields onto settings
            for field in bootstrap.model_fields:
                if hasattr(_settings, field):
                    setattr(_settings, field, getattr(bootstrap, field))
            
            loader = get_app_config_loader()
            if loader:
                logger.info("Loading config from Azure App Configuration")
                # Trigger an initial refresh check
                try:
                    loader.refresh()
                except Exception as e:
                    logger.warning(f"Initial refresh check failed: {e}")
                _populate_from_app_config(_settings, loader)
            elif bootstrap.use_key_vault and bootstrap.azure_key_vault_url:
                logger.info("Loading secrets from Azure Key Vault")
                kv = KeyVaultLoader(bootstrap.azure_key_vault_url)
                for setting_name in KeyVaultLoader.SECRET_MAPPING:
                    value = kv.get_secret(setting_name)
                    if value:
                        setattr(_settings, setting_name, value)
            else:
                logger.info("Loading config from environment variables")
            
            # Validate
            if not _settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required")
        
        # Subsequent calls: refresh from App Config if enabled
        else:
            loader = get_app_config_loader()
            if loader:
                try:
                    loader.refresh()  # No-op if interval hasn't elapsed
                    _populate_from_app_config(_settings, loader)
                except Exception as e:
                    logger.warning(f"App Config refresh failed: {e}")
        
        return _settings


def refresh_settings() -> Settings:
    """Force an immediate refresh and return new settings."""
    loader = get_app_config_loader()
    if loader:
        loader.refresh()
    return get_settings()
