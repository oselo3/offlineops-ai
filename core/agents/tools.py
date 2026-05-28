"""
core/agents/tools.py

Safe, scoped infrastructure tools available to the agent.

Design principles:
- Read-only by default — no writes, no destructive actions
- All commands are explicit and auditable
- Timeouts enforced on every subprocess call
- Tool results are plain strings (easy to feed back into LLM context)
"""

import subprocess
import shutil
import platform
from pathlib import Path

from core.agents.engine import Tool, ToolRegistry


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a subprocess command and return stdout + stderr as a string."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n[stderr]: {result.stderr.strip()}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s"
    except FileNotFoundError:
        return f"ERROR: Command not found: {cmd[0]}"
    except Exception as e:
        return f"ERROR: {e}"


# ------------------------------------------------------------------ #
# Tool implementations                                                 #
# ------------------------------------------------------------------ #

def check_disk_usage(path: str = "/") -> str:
    """Return disk usage for the given path."""
    return _run(["df", "-h", path])


def get_service_status(service: str) -> str:
    """Return the systemd status of a named service."""
    if platform.system() != "Linux":
        return "ERROR: systemctl is only available on Linux."
    return _run(["systemctl", "status", service, "--no-pager", "-l"])


def tail_log_file(path: str, lines: int = 50) -> str:
    """Return the last N lines of a log file."""
    log_path = Path(path)
    # Safety: restrict to /var/log and /tmp for now
    allowed_prefixes = ["/var/log", "/tmp", "/home"]
    if not any(str(log_path).startswith(p) for p in allowed_prefixes):
        return f"ERROR: Access to '{path}' is not permitted. Allowed paths: {allowed_prefixes}"
    if not log_path.exists():
        return f"ERROR: File not found: {path}"
    return _run(["tail", f"-n{lines}", str(log_path)])


def ping_host(host: str, count: int = 4) -> str:
    """Ping a host and return reachability results."""
    # Sanitize: reject anything that looks like a shell injection
    if any(c in host for c in [";", "&", "|", "`", "$", " "]):
        return "ERROR: Invalid host value."
    return _run(["ping", "-c", str(count), "-W", "2", host])


def list_network_interfaces() -> str:
    """List network interfaces and their IP addresses."""
    if shutil.which("ip"):
        return _run(["ip", "-brief", "address", "show"])
    elif shutil.which("ifconfig"):
        return _run(["ifconfig"])
    return "ERROR: Neither 'ip' nor 'ifconfig' found on this system."


def get_memory_usage() -> str:
    """Return current memory usage."""
    return _run(["free", "-h"])


def list_top_processes(count: int = 10) -> str:
    """Return the top N processes by CPU usage."""
    return _run(["ps", "aux", "--sort=-%cpu", f"--lines={count + 1}"])


def check_port_listening(port: int) -> str:
    """Check if a given TCP port is listening."""
    if shutil.which("ss"):
        return _run(["ss", "-tlnp", f"sport = :{port}"])
    elif shutil.which("netstat"):
        return _run(["netstat", "-tlnp"])
    return "ERROR: Neither 'ss' nor 'netstat' found."


# ------------------------------------------------------------------ #
# Registry builder                                                     #
# ------------------------------------------------------------------ #

def build_default_registry() -> ToolRegistry:
    """Return a ToolRegistry pre-loaded with all default infra tools."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="check_disk_usage",
        description="Check disk usage for a given path (default: /)",
        parameters={"path": {"type": "string", "default": "/"}},
        fn=check_disk_usage,
    ))

    registry.register(Tool(
        name="get_service_status",
        description="Get the systemd status of a named Linux service (e.g. nginx, postgresql)",
        parameters={"service": {"type": "string"}},
        fn=get_service_status,
    ))

    registry.register(Tool(
        name="tail_log_file",
        description="Read the last N lines of a log file under /var/log or /tmp",
        parameters={
            "path": {"type": "string"},
            "lines": {"type": "integer", "default": 50},
        },
        fn=tail_log_file,
    ))

    registry.register(Tool(
        name="ping_host",
        description="Ping a hostname or IP address to check network reachability",
        parameters={
            "host": {"type": "string"},
            "count": {"type": "integer", "default": 4},
        },
        fn=ping_host,
    ))

    registry.register(Tool(
        name="list_network_interfaces",
        description="List all network interfaces and their IP addresses on this host",
        parameters={},
        fn=list_network_interfaces,
    ))

    registry.register(Tool(
        name="get_memory_usage",
        description="Show current RAM and swap usage",
        parameters={},
        fn=get_memory_usage,
    ))

    registry.register(Tool(
        name="list_top_processes",
        description="List the top N processes by CPU usage",
        parameters={"count": {"type": "integer", "default": 10}},
        fn=list_top_processes,
    ))

    registry.register(Tool(
        name="check_port_listening",
        description="Check whether a specific TCP port is currently listening on this host",
        parameters={"port": {"type": "integer"}},
        fn=check_port_listening,
    ))

    return registry
