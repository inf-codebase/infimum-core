from .package_utils import ensure_package_installed, install_package
from core.base import create_lazy_getattr

__all__ = [
    "ensure_package_installed",
    "install_package",
    # Security submodule exports (available via core.engine.security)
    "security",
]

# Lazy imports to avoid loading heavy dependencies (database, etc.) when not needed
__getattr__ = create_lazy_getattr(
    __name__,
    decorator_map={"singleton": ".decorators"},
    submodules=["context", "startup", "security"]
)