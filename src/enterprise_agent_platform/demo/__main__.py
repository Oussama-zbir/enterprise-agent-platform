"""``python -m enterprise_agent_platform.demo`` — the end-to-end demo.

Exits non-zero if the platform no longer does what the transcript claims.
"""

from __future__ import annotations

import asyncio
import sys

from enterprise_agent_platform.demo.walkthrough import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
