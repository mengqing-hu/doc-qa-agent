"""Centralized configuration access for the RAG pipeline."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when project configuration is missing or invalid."""


class Config:
    """Load project settings and expose common RAG configuration values."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        env_path: str | Path = DEFAULT_ENV_PATH,
    ) -> None:
        """Load configuration from the supplied project files.

        Args:
            config_path: Path to the YAML configuration file.
            env_path: Path to the dotenv file.
        """
        # Configuration file paths
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._config = load_config(self.config_path, env_path=self.env_path)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Return a nested setting, or ``default`` when its path is absent.

        Args:
            *keys: Ordered keys describing a path through the configuration.
            default: Value to return when the path does not exist.

        Returns:
            The configured value or the supplied default.
        """
        # Traverse nested configuration values
        value: Any = self._config
        for key in keys:
            # Stop early so callers can choose a safe fallback value.
            if not isinstance(value, Mapping) or key not in value:
                return default
            value = value[key]
        return value

    # API credentials
    @property
    def scadsai_api_key(self) -> str:
        """Return the ScaDS.AI API key required for LLM requests."""
        return get_required_environment_variable("SCADS_API_KEY")

    @property
    def hf_token(self) -> str:
        """Return the optional Hugging Face token when a caller requires it."""
        return get_required_environment_variable("HF_TOKEN")

    # Project paths
    @property
    def log_dir(self) -> Path:
        """Return the configured directory for application logs."""
        return Path(self.get("logging", "log_dir", default="logs"))


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> dict[str, Any]:
    """Load YAML configuration after making dotenv values available.

    Args:
        config_path: Path to the YAML configuration file.
        env_path: Path to the dotenv file. Existing environment values win.

    Returns:
        The resolved configuration mapping.
    """
    config_file = Path(config_path)
    dotenv_file = Path(env_path)

    # Load environment variables
    # Runtime-injected values take precedence over local dotenv defaults.
    logger.debug("Loading configuration from %s", config_file)
    _load_dotenv_file(dotenv_file)

    # Load YAML configuration
    config_data = _load_yaml_mapping(config_file)

    # Resolve environment placeholders
    resolved_config = _resolve_environment_variables(config_data)
    logger.debug("Loaded %d top-level configuration section(s)", len(resolved_config))
    return resolved_config


def get_required_environment_variable(variable_name: str) -> str:
    """Return a non-empty environment variable or raise a clear error.

    Args:
        variable_name: Name of the required environment variable.

    Returns:
        The environment variable value.
    """
    value = os.getenv(variable_name)
    if value is None or not value.strip():
        logger.error("Required environment variable is not configured: %s", variable_name)
        raise ConfigurationError(
            f"Required environment variable '{variable_name}' is not configured. "
            "Set it in .env or the runtime environment."
        )
    return value


def _load_dotenv_file(env_path: Path) -> None:
    """Load dotenv values without overriding runtime configuration."""
    if env_path.exists() and not env_path.is_file():
        logger.error("Dotenv path is not a file: %s", env_path)
        raise ConfigurationError(f"Dotenv path is not a file: {env_path}")

    # Load local development defaults
    logger.debug("Loading dotenv values from %s", env_path)
    load_dotenv(env_path, override=False)


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """Read a YAML file and ensure its root value is a mapping."""
    if not config_path.is_file():
        logger.error("Configuration file was not found: %s", config_path)
        raise ConfigurationError(f"Configuration file was not found: {config_path}")

    # Parse YAML safely
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        logger.error("Configuration file contains invalid YAML: %s", config_path)
        raise ConfigurationError(
            f"Configuration file contains invalid YAML: {config_path}"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        logger.error("Configuration file could not be read: %s", config_path)
        raise ConfigurationError(
            f"Configuration file could not be read: {config_path}"
        ) from error

    # Validate configuration structure
    if config_data is None:
        return {}
    if not isinstance(config_data, Mapping):
        logger.error("Configuration root must be a mapping: %s", config_path)
        raise ConfigurationError(
            f"Configuration root must be a mapping: {config_path}"
        )
    return dict(config_data)


def _resolve_environment_variables(value: Any) -> Any:
    """Recursively replace ${VARIABLE} references in YAML values."""
    if isinstance(value, Mapping):
        return {
            key: _resolve_environment_variables(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_environment_variables(item) for item in value]
    if isinstance(value, str):
        # Resolve placeholders in nested URLs, headers, and list entries.
        return ENVIRONMENT_VARIABLE_PATTERN.sub(
            _replace_environment_variable,
            value,
        )
    return value


def _replace_environment_variable(match: re.Match[str]) -> str:
    """Resolve one environment-variable placeholder in a string value."""
    return get_required_environment_variable(match.group(1))
