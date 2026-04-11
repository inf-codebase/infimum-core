from __future__ import annotations

"""
Utilities for discovering and registering FastAPI routers in reusable core packages.

This module does not import FastAPI at load time; FastAPI is only required when you
call register_package (discovery), include_all, or use with_registered_routers. Install
FastAPI in your project (or use the optional extra: pip install infimum-core[api]).

Typical usage in a downstream FastAPI app:

    from fastapi import FastAPI
    from infimum.base.api_router_registry import APIRouterRegistry

    app = FastAPI(
        title="Project Infimum",
        version="0.1.0",
        debug=True,
        lifespan=lifespan,
        swagger_ui_parameters={
            "docExpansion": "none",
            "defaultModelsExpandDepth": -1,
        },
    )

    # Register all routers from one or more API packages
    APIRouterRegistry.register_package("core.base.user_management.api")
    # APIRouterRegistry.register_package("core.some_other_feature.api")

    # Mount everything under a common prefix, e.g. /api/v1
    APIRouterRegistry.include_all(app, prefix="/api/v1")

Decorator usage (register packages and mount in one place):

    from infimum.base.api_router_registry import with_registered_routers

    @with_registered_routers(
        packages=["core.base.user_management.api"],
        prefix="/api/v1",
    )
    def create_app() -> FastAPI:
        return FastAPI(title="Project Infimum", ...)
"""

from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional
import importlib
import pkgutil

from loguru import logger

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI


def _get_apirouter() -> type:
    """Lazy import so FastAPI is only required when using router discovery."""
    try:
        from fastapi import APIRouter as _APIRouter
        return _APIRouter
    except ImportError as e:
        raise RuntimeError(
            "FastAPI is required for API router discovery and mounting. "
            "Install it in your project: pip install fastapi"
        ) from e


def _discover_routers_from_package(package_path: str) -> List[Any]:
    """
    Discover all FastAPI APIRouter instances exposed as `router` attributes
    in the direct submodules of the given package.

    Convention:
    - `package_path` points to a Python package (e.g. "core.base.user_management.api")
    - Each submodule of that package (e.g. "admin_controller", "auth_controller")
      may define a top-level variable: `router = APIRouter(...)`
    - Only attributes that are instances of fastapi.APIRouter are returned.
    """
    try:
        package = importlib.import_module(package_path)
    except ImportError as e:
        logger.error(f"Could not import API package '{package_path}': {e}")
        return []

    if not hasattr(package, "__path__"):
        # Not a package, nothing to discover
        logger.warning(f"Path '{package_path}' is not a package; skipping router discovery")
        return []

    APIRouter = _get_apirouter()
    discovered: List[Any] = []

    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module_name = module_info.name
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.error(f"Failed to import API module '{module_name}': {e}")
            continue

        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            discovered.append(router)
            logger.debug(f"Discovered router in '{module_name}'")
        else:
            logger.debug(f"No APIRouter named 'router' in '{module_name}'")

    logger.info(f"Discovered {len(discovered)} routers from package '{package_path}'")
    return discovered


def with_registered_routers(
    *,
    packages: Iterable[str],
    prefix: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for FastAPI app factory functions.

    Registers all routers from the given packages and mounts them on the app
    returned by the factory, with an optional global prefix.

    Example:
        @with_registered_routers(
            packages=["core.base.user_management.api"],
            prefix="/api/v1",
        )
        def create_app() -> FastAPI:
            return FastAPI(title="Project Infimum", ...)
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: object, **kwargs: object) -> Any:
            app = fn(*args, **kwargs)
            for pkg in packages:
                APIRouterRegistry.register_package(pkg)
            APIRouterRegistry.include_all(app, prefix=prefix)
            return app
        return wrapper
    return decorator


class APIRouterRegistry:
    """
    Registry for FastAPI APIRouter instances, similar in spirit to EntityRegistry.

    Supports:
    - Explicit router registration via `register_router`
    - Package-based discovery via `register_package`
    - Mounting all registered routers onto a FastAPI app via `include_all`

    FastAPI is not a dependency of core; install it in your project when using this registry.
    """

    _routers: List[Any] = []
    _router_ids: set[int] = set()

    @classmethod
    def register_router(cls, router: Any) -> None:
        """
        Register a single APIRouter instance.

        The same router will only be registered once, even if called multiple times.
        """
        if router is None:
            return

        router_id = id(router)
        if router_id in cls._router_ids:
            logger.debug("Router already registered; skipping duplicate")
            return

        cls._routers.append(router)
        cls._router_ids.add(router_id)
        logger.debug(f"Registered APIRouter: {getattr(router, 'prefix', '')} "
                     f"tags={getattr(router, 'tags', None)}")

    @classmethod
    def register_package(cls, package_path: str) -> None:
        """
        Discover and register all APIRouter instances from the given package.

        Example:
            APIRouterRegistry.register_package("core.base.user_management.api")
        """
        routers = _discover_routers_from_package(package_path)
        for router in routers:
            cls.register_router(router)

    @classmethod
    def include_all(cls, app: Any, prefix: Optional[str] = None) -> None:
        """
        Include all registered routers into the given FastAPI app.

        If `prefix` is provided, all routers will be mounted under that prefix:
            APIRouterRegistry.include_all(app, prefix="/api/v1")
        """
        if not cls._routers:
            logger.warning("APIRouterRegistry.include_all called with no registered routers")
            return

        for router in cls._routers:
            if prefix:
                app.include_router(router, prefix=prefix)
                logger.debug(
                    f"Included router with prefix '{getattr(router, 'prefix', '')}' "
                    f"under global prefix '{prefix}'"
                )
            else:
                app.include_router(router)
                logger.debug(
                    f"Included router with prefix '{getattr(router, 'prefix', '')}'"
                )

        logger.info(f"Included {len(cls._routers)} routers into FastAPI app")

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered routers.

        Useful for tests or re-initialization scenarios.
        """
        cls._routers.clear()
        cls._router_ids.clear()
        logger.debug("Cleared APIRouterRegistry")

