"""
Configuration manager for loading and validating config.yaml.

Provides centralized configuration access with validation to ensure
all settings are correct before tournament execution.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import yaml

from src.data_models import (
    Config, ModelConfig, ScoringConfig, TrainingConfig,
    DeviceConfig, BradleyTerryConfig
)


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manager for tournament configuration.
    
    Loads config.yaml, validates all settings, and provides typed
    access to configuration sections.
    
    Attributes:
        config_path: Path to config.yaml file
        config: Loaded and validated Config object
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config_path is invalid
        """
        if not config_path:
            raise ValueError("config_path cannot be empty")
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        self.config: Config = None
        
        logger.info(f"ConfigManager initialized with: {self.config_path}")
    
    def load_config(self) -> Config:
        """
        Load and validate configuration from config.yaml.
        
        Returns:
            Validated Config object
        
        Raises:
            RuntimeError: If config loading or validation fails
        """
        try:
            logger.info(f"Loading configuration from: {self.config_path}")
            
            # Load YAML file
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ValueError("Configuration file is empty")
            
            # Parse and validate configuration
            self.config = self._parse_config(config_data)
            
            logger.info("Configuration loaded and validated successfully")
            
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise RuntimeError(f"YAML parsing failed: {e}") from e
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e
    
    def _parse_config(self, config_data: Dict[str, Any]) -> Config:
        """
        Parse and validate configuration data.
        
        Args:
            config_data: Raw configuration dictionary from YAML
        
        Returns:
            Validated Config object
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Parse models
        models_data = config_data.get('models', {})
        if not models_data:
            raise ValueError("Configuration must include 'models' section")
        
        models = {}
        for model_name, model_config in models_data.items():
            # Use .get() so a missing hf_id yields ModelConfig's friendly
            # ValueError (with the expected 'org/model-name' format) instead of
            # an opaque KeyError.
            models[model_name] = ModelConfig(
                hf_id=model_config.get('hf_id'),
                quantization=model_config.get('quantization', 'fp16')
            )
        
        # Parse scoring
        scoring_data = config_data.get('scoring', {})
        if not scoring_data:
            raise ValueError("Configuration must include 'scoring' section")
        
        weights = scoring_data.get('weights', {})
        if not weights:
            raise ValueError("Scoring section must include 'weights'")
        
        bt_mle_data = scoring_data.get('bt_mle', {})
        bt_mle = BradleyTerryConfig(
            beta=float(bt_mle_data.get('beta', 1e-10))
        )
        
        scoring = ScoringConfig(
            weights=weights,
            bt_mle=bt_mle
        )
        
        # Parse training
        training_data = config_data.get('training', {})
        if not training_data:
            raise ValueError("Configuration must include 'training' section")
        
        training = TrainingConfig(
            checkpoint_every=int(training_data.get('checkpoint_every', 100)),
            batch_size=int(training_data.get('batch_size', 1)),
            learning_rate=float(training_data.get('learning_rate', 0.001))
        )
        
        # Parse device
        device_data = config_data.get('device', {})
        device = DeviceConfig(
            auto_detect=device_data.get('auto_detect', True),
            fallback=device_data.get('fallback', 'cpu')
        )
        
        # Create Config object (validation happens in __post_init__)
        config = Config(
            models=models,
            scoring=scoring,
            training=training,
            device=device
        )
        
        return config
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            ModelConfig for the requested model
        
        Raises:
            ValueError: If model not found in configuration
            RuntimeError: If config not loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        if model_name not in self.config.models:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {list(self.config.models.keys())}"
            )
        
        return self.config.models[model_name]
    
    def get_scorer_weight(self, scorer_name: str) -> float:
        """
        Get weight for a specific scorer.
        
        Args:
            scorer_name: Name of the scorer
        
        Returns:
            Weight value for the scorer
        
        Raises:
            ValueError: If scorer not found in configuration
            RuntimeError: If config not loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        if scorer_name not in self.config.scoring.weights:
            raise ValueError(
                f"Scorer '{scorer_name}' not found in configuration. "
                f"Available scorers: {list(self.config.scoring.weights.keys())}"
            )
        
        return self.config.scoring.weights[scorer_name]
    
    def validate_scorer_weights(self) -> bool:
        """
        Validate that scorer weights sum to 1.0.
        
        Returns:
            True if weights are valid
        
        Raises:
            RuntimeError: If config not loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        # Validation already happens in ScoringConfig.__post_init__
        # This method just provides explicit validation access
        return True
    
    def get_raw_config(self) -> Dict[str, Any]:
        """
        Get raw configuration as dictionary.
        
        Returns:
            Dictionary representation of configuration
        
        Raises:
            RuntimeError: If config not loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        # Convert Config object back to dictionary
        return {
            'models': {
                name: {'hf_id': cfg.hf_id, 'quantization': cfg.quantization}
                for name, cfg in self.config.models.items()
            },
            'scoring': {
                'weights': self.config.scoring.weights,
                'bt_mle': {'beta': self.config.scoring.bt_mle.beta}
            },
            'training': {
                'checkpoint_every': self.config.training.checkpoint_every,
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.learning_rate
            },
            'device': {
                'auto_detect': self.config.device.auto_detect,
                'fallback': self.config.device.fallback
            }
        }
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_path='{self.config_path}')"
