import random
from typing import List
from decouple import AutoConfig
import ast
import sys
import os
from string import Template

class AutoConfigImpl(AutoConfig):
    """
    AutoConfig implementation that handles type conversion and environment variable fallback.

    * Case 1: If .env file doesn't exist, the system will still work and check system environment variables when accessed
    * Case 2: If .env exists but a specific variable is missing, when that variable is accessed, it will automatically fall back to checking the system environment
    * Fallback pattern: Any missing variable will default to "{VARIABLE_NAME}_IS_NOT_SET" if not found in system environment

    Args:
        search_path (str, optional): Path to search for .env file. Defaults to None.
    """

    def __init__(self, search_path=None):
        super(AutoConfigImpl, self).__init__(search_path)
        self.search_path = search_path
        self._load(self.search_path or self._caller_path())

        # Handle case where repository is empty (no .env file)
        if not hasattr(self.config.repository, 'data'):
            # Create an empty data dict for RepositoryEmpty
            self.config.repository.data = {}

        # verify "type ="
        env_context = os.environ.copy()
        env_context.update({k: str(v) for k, v in self.config.repository.data.items()})
        for option, value in self.config.repository.data.items():
            raw_value = str(value).strip()
            expanded_value = Template(raw_value).safe_substitute(env_context)

            if ',type=' in expanded_value:
                base_value, type_hint = expanded_value.split(',type=', 1)
                base_value = base_value.strip()
                type_hint = type_hint.strip()

                if type_hint == 'int':
                    parsed_value = int(base_value)
                elif type_hint in ('list', 'dict'):
                    parsed_value = ast.literal_eval(base_value)
                elif type_hint == 'bool':
                    parsed_value = base_value == 'True'
                else:
                    parsed_value = base_value
            else:
                parsed_value = expanded_value

            self.config.repository.data[option] = parsed_value
            env_context[option] = str(parsed_value)


env_file = './.env'
# Remove assertion and handle missing .env file gracefully
config = None
if os.path.exists(env_file):
    config = AutoConfigImpl(env_file)
else:
    # Create a config even when .env doesn't exist
    config = AutoConfigImpl()

this_module = sys.modules[__name__]
for option, value in config.config.repository.data.items():
    setattr(this_module, option, value)
    os.environ[option] = str(value)

# Add module-level __getattr__ to handle missing variables by checking system environment
def __getattr__(name):
    """
    If an attribute is not found in the module, try to get it from system environment.
    This handles cases where variables are missing from .env file.
    """
    value = os.getenv(name, None)
    # Cache the value in the module for future access
    setattr(this_module, name, value)
    return value

def get_random_config(config_values):
    if isinstance(config_values, list) or isinstance(config_values, tuple):
        return config_values[random.randint(0, len(config_values) - 1)]
    else:
        return config_values

def get_config_by_prefix(prefix: str):
    return {k: v for k, v in config.config.repository.data.items() if k.startswith(prefix)}

def get_config_by_prefixes(prefixes: List[str]):
    return {k: v for k, v in config.config.repository.data.items() if any(k.startswith(prefix) for prefix in prefixes)}


# ============================================================================
# GPU Detection - Centralized Configuration
# ============================================================================

def _detect_gpu_config():
    """
    Detect GPU availability and configuration using PyTorch.

    Returns:
        tuple: (has_cuda: bool, device_count: int, cuda_version: str or None, has_mps: bool)
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if has_cuda else 0
        cuda_version = torch.version.cuda if has_cuda else None

        # Check for MPS (Metal Performance Shaders) on Mac
        has_mps = False
        if not has_cuda:  # Only check MPS if no CUDA
            try:
                has_mps = torch.backends.mps.is_available()
                if has_mps:
                    # Test if MPS actually works
                    torch.zeros(1, device="mps")
            except Exception:
                has_mps = False

        return has_cuda, device_count, cuda_version, has_mps
    except ImportError:
        # PyTorch not installed - assume CPU only
        return False, 0, None, False
    except Exception as e:
        # Any other error - fallback to CPU
        print(f"[auto_config] GPU detection failed: {e}")
        return False, 0, None, False


# Initialize GPU configuration at module load time
CUDA_AVAILABLE, GPU_COUNT, CUDA_VERSION, MPS_AVAILABLE = _detect_gpu_config()

# Convenience attributes
HAS_GPU = CUDA_AVAILABLE
NUM_GPUS = GPU_COUNT
HAS_MPS = MPS_AVAILABLE

# Determine if we have any hardware acceleration
HAS_ACCELERATION = CUDA_AVAILABLE or MPS_AVAILABLE
