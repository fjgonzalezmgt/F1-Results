"""Loguru setup and lightweight function instrumentation."""

from __future__ import annotations

import functools
import inspect
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar, cast

from loguru import logger

from f1predictor.config import RESULTS_DIR


F = TypeVar("F", bound=Callable[..., Any])
LOG_PATH = RESULTS_DIR / "app.log"

_CONFIGURED = False


def configure_logging(log_path: Path = LOG_PATH) -> Path:
    """Configure Loguru sinks once for the app process."""
    global _CONFIGURED
    if _CONFIGURED:
        return log_path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    console_level = os.getenv("F1PREDICTOR_LOG_LEVEL", "INFO").upper()
    file_level = os.getenv("F1PREDICTOR_FILE_LOG_LEVEL", "DEBUG").upper()
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{name}:{function}:{line} - {message}"
    )

    logger.remove()
    logger.add(sys.stderr, level=console_level, format=log_format)
    logger.add(
        log_path,
        level=file_level,
        format=log_format,
        rotation="5 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True
    logger.info(
        "Logging inicializado. Consola={}, archivo={}, ruta={}",
        console_level,
        file_level,
        log_path,
    )
    return log_path


def log_call(func: F | None = None, *, level: str = "INFO") -> F | Callable[[F], F]:
    """Log function entry, exit elapsed time and exceptions."""

    def decorate(target: F) -> F:
        if getattr(target, "_f1predictor_logged", False):
            return target

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            configure_logging()
            call_label = f"{target.__module__}.{target.__qualname__}"
            call_summary = _summarize_call(args, kwargs)
            started = time.perf_counter()
            logger.log(level, "Inicia {}({})", call_label, call_summary)
            try:
                result = target(*args, **kwargs)
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.exception("Error en {} despues de {:.1f} ms", call_label, elapsed_ms)
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.log(level, "Termina {} en {:.1f} ms", call_label, elapsed_ms)
            return result

        setattr(wrapper, "_f1predictor_logged", True)
        return cast(F, wrapper)

    if func is None:
        return decorate
    return decorate(func)


def instrument_module_functions(
    module_name: str,
    *,
    skip: Iterable[str] = (),
    public_level: str = "INFO",
    private_level: str = "DEBUG",
) -> None:
    """Wrap all top-level functions in a module with ``log_call``."""
    configure_logging()
    module = sys.modules[module_name]
    skipped = set(skip)
    for name, value in list(vars(module).items()):
        if name in skipped:
            continue
        if not inspect.isfunction(value) or value.__module__ != module_name:
            continue
        level = private_level if name.startswith("_") else public_level
        setattr(module, name, log_call(value, level=level))


def _summarize_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in args[:3]:
        parts.append(_summarize_value(value))
    if len(args) > 3:
        parts.append(f"...+{len(args) - 3} args")
    for key, value in list(kwargs.items())[:3]:
        parts.append(f"{key}={_summarize_value(value)}")
    if len(kwargs) > 3:
        parts.append(f"...+{len(kwargs) - 3} kwargs")
    return ", ".join(parts)


def _summarize_value(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple):
        return f"{type(value).__name__}(shape={shape})"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        clean = value.replace("\n", "\\n")
        return repr(clean[:80] + ("..." if len(clean) > 80 else ""))
    if isinstance(value, (int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__
