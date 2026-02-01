import subprocess
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TerminalTools:
    """
    Terminal execution limb with safeguards.
    """
    def __init__(self, workspace_root: str, enabled: bool = False):
        self.workspace_root = os.path.abspath(workspace_root)
        self.enabled = enabled
        # Basic blocklist for "dumb" safeguards. 
        # Real security requires containerization.
        self.blocked_commands = ["sudo", "rm -rf", "mkfs", ":(){ :|:& };:"] 

    def run_command(self, command: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute a shell command.
        
        Args:
            command: The command string to run.
            timeout: Max seconds to wait.
            
        Returns:
            Dict with 'stdout', 'stderr', 'returncode', or 'error'.
        """
        if not self.enabled:
            return {"status": "error", "message": "Terminal access is disabled via configuration."}

        # 1. Safeguard: Blocklist
        for blocked in self.blocked_commands:
            if blocked in command:
                return {
                    "status": "error", 
                    "message": f"Command contains forbidden pattern: '{blocked}'"
                }

        # 2. Execution
        try:
            logger.info(f"Executing terminal command: {command}")
            # shell=True allows pipes and natural syntax, but carries injection risk.
            # Given this is an "Autodidact" running local code, we assume the Brain 
            # is the user, but we enforce CWD.
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "status": "success",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
