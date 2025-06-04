"""Simple alert system used by tests."""

import logging
from datetime import datetime


class AlertSystem:
    """In-memory alert dispatcher for unit tests."""

    def __init__(self):
        self.history = []
        self.logger = logging.getLogger("AlertSystem")

    def send_alert(self, message, level="info"):
        """Record an alert and print it to the log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        self.history.append(entry)

        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def get_history(self):
        """Return list of all alerts sent."""
        return list(self.history)
