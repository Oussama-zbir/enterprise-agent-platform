"""A runnable demonstration of the platform, with no key and no network.

The platform ships no tools of its own: a deployment declares what its agents
may do. That is the right default and it leaves a reader with nothing to run,
so this package is one deployment's tool module — synthetic accounts-payable
data, four tools spanning the risk range, and a scripted stand-in for a model —
and a walkthrough that drives the whole approval cycle through the HTTP API.

Nothing here is imported by the platform's own code paths except through
configuration (``EAP_LLM_PROVIDER=demo``, ``EAP_DEMO_TOOLS=true``), both of
which are refused in production.
"""

from enterprise_agent_platform.demo.data import Ledger
from enterprise_agent_platform.demo.provider import DemoLLMProvider
from enterprise_agent_platform.demo.tools import build_demo_tools

__all__ = ["DemoLLMProvider", "Ledger", "build_demo_tools"]
