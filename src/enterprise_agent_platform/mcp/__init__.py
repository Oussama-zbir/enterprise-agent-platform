"""Model Context Protocol client.

Tools published by an external MCP server are adapted into the platform's own
``Tool`` type and registered in the ordinary ``ToolRegistry``, so the agent
runner, the risk gate, and the approval workflow never learn that a capability
came from somewhere else. The dependency points one way: this package imports
the tool layer, nothing imports this package except the wiring that builds an
application.
"""
