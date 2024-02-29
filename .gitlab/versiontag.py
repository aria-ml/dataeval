from datetime import date

from gitlab import Gitlab
from verboselog import verbose

# fmt: off
# Bi-weekly sprints starting 8/2/2023
MINOR_VERSION_SPRINT_MAPPING = [
    21, 22, 23, 24, 25,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46,
    51, 52, 53, 54, 55, 56,
    61, 62, 63, 64, 65, 66,
]
# fmt: on


class VersionTag:
    """
    Provides current and pending/next version number for DAML
    """

    def __init__(self, gitlab: Gitlab):
        self.gl = gitlab
        self._current = None
        self._pending = None

    @property
    def current(self) -> str:
        """
        The current version of DAML retrieved from repository tags
        """
        if self._current is None:
            tags = self.gl.list_tags()
            verbose(f"Pulled tags: {', '.join(d['name'] for d in tags)}")
            for tag in tags:
                if len(tag["name"].split(".")) != 3:
                    continue
                self._current = tag["name"]
                break
            if self._current is None:
                raise ValueError("Unable to get current version.")
        return self._current

    @property
    def pending(self) -> str:
        """
        The calculated next version based off of calendar and sprint cycle
        """
        if self._pending is None:
            major, minor, patch = self.current.split(".")
            sprint = int((date.today() - date(2023, 8, 2)).days / 14)
            pending_minor = MINOR_VERSION_SPRINT_MAPPING[sprint]
            verbose(f"Current: {int(minor)} Pending: {pending_minor}")
            if int(minor) == pending_minor:
                version = f"{major}.{minor}.{int(patch) + 1}"
            else:
                version = f"{major}.{pending_minor}.0"
            self._pending = version
        return self._pending
