from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("The Greatest Athlete of All Time — pipeline entry point")
    # Batch scoring pipeline entry point.
    # Interactive dashboard: streamlit run dashboards/app.py
    # API server:            uvicorn src.api.main:app --reload
    # Tests:                 pytest tests/ -v --cov=src


if __name__ == "__main__":
    main()
