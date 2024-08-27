from typing import Literal

from gitlab import Gitlab
from rest import verbose


class VersionTag:
    """
    Provides current and pending/next version number for DataEval
    """

    def __init__(self, gitlab: Gitlab):
        self.gl = gitlab
        self._current = None
        self._pending = None

    @property
    def current(self) -> str:
        """
        The current version of DataEval retrieved from repository tags
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

    def next(self, version_type: Literal["MAJOR", "MINOR", "PATCH"]):
        current = self.current
        version = current
        major, minor, patch = current.split(".")
        if version_type == "PATCH":
            pending_patch = str(int(patch) + 1)
            version = f"{major}.{minor}.{pending_patch}"
        elif version_type == "MINOR":
            pending_minor = str(int(minor) + 1)
            version = f"{major}.{pending_minor}.0"
        elif version_type == "MAJOR":  # 6
            # strip off the 'v' add 1 and add the v back in.
            temp = major[1:]  # make sure to hand 1+ digits.
            pending_major = "v" + str(int(temp) + 1)
            version = f"{pending_major}.0.0"

        verbose(f"Bumping version from {self._current} to {version}, change is {version_type}")
        return version
