import inspect
import json
import logging

from rllm.environments.swe.swe import SWEEnv

logger = logging.getLogger(__name__)


class CLIEnv(SWEEnv):
    """CLI Agent training environment.

    Extends SWEEnv with CLI-specific capabilities:
    - Optional CLAUDE.md-style context file injection into the container
    - Same Docker lifecycle, tools, reward computation as SWE
    """

    def __init__(self, context_file: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.context_file = context_file

    def reset(self) -> tuple[str, dict]:
        obs, info = super().reset()
        if self.context_file:
            self._inject_context_file()
        return obs, info

    def _inject_context_file(self):
        """Write the context file content into /testbed/CLAUDE.md inside the container."""
        if self.env is None or self.env.runtime is None:
            logger.warning("Cannot inject context file: environment runtime not available")
            return

        escaped = self.context_file.replace("'", "'\\''")
        cmd = f"cat > /testbed/CLAUDE.md << 'CTXEOF'\n{escaped}\nCTXEOF"
        output, error_code = self.env.runtime.run(cmd, timeout=15)
        if error_code and "Error" in str(error_code):
            logger.warning("Failed to inject context file: %s", output)

    @staticmethod
    def from_dict(extra_info: dict | str) -> "CLIEnv":
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(CLIEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return CLIEnv(**init_params)
