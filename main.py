"""Точка входа для демонстрационного приложения проекта isnews."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _is_running_inside_streamlit() -> bool:
    """Проверяет, запущен ли текущий процесс через Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    return get_script_run_ctx() is not None


def _run_streamlit() -> int:
    """Запускает текущий файл как Streamlit-приложение."""
    script_path = Path(__file__).resolve()
    command = [sys.executable, "-m", "streamlit", "run", str(script_path)]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _render_application() -> None:
    """Отрисовывает интерфейс приложения, если код выполняется внутри Streamlit."""
    from src.isnews.ui import render_main_page

    render_main_page()


def main() -> int:
    """Выбирает подходящий режим запуска приложения."""
    if _is_running_inside_streamlit():
        _render_application()
        return 0

    return _run_streamlit()


if __name__ == "__main__":
    raise SystemExit(main())
