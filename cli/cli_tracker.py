from dataclasses import dataclass


@dataclass
class CliTracker:
    """click context object"""

    verbose: int = 1
