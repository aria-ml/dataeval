# Configuration file for IPython Kernel to silence noisy warnings.
try:
    c = get_config()
except NameError:
    from traitlets.config import Config

    c = Config()

# Silence the "Kernel is running over TCP without encryption" warning and other non-critical startup logs.
c.IPKernelApp.log_level = "CRITICAL"
