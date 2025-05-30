import yaml
from pathlib import Path
from typing import Dict, Any

class Configs:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

    @staticmethod
    def merge_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        merged_config = config1.copy()
        for key, value in config2.items():
            if key not in merged_config:
                merged_config[key] = value
            elif isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key] = Configs.merge_configs(merged_config[key], value)
        return merged_config
