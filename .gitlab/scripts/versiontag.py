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

    @property
    def pending(self) -> str:
        """
        Release value is structured as "Major.Minor.Patch"
        Calculate the next version based off of the kind of change(s) since the last release.  Largest change wins
        For example, a FIX or IMPROVEMENT leads to a Patch increment but a new FEATURE or DEPRECATION of an old feature
        leads to a Minor version increase and a Major change increments the Major version number.
        """

        if self._pending is None:
            verbose("Getting a pending value before it has been set")
            return "0.0.0"
        else:
            verbose(f"Getting a new pending value {self._pending}")
            return self._pending  # if it's been set from the outside return it.

    # add a setter so it can be set by commitgen.
    @pending.setter
    def pending(self, value):
        verbose(f"Setting a new pending value. Value is {value}")
        self._pending = value

    def next(self, version_type: Literal["MAJOR", "MINOR", "PATCH"]):
        current = self.current
        version = current
        major, minor, patch = current.split(".")
        if version_type == "PATCH":
            pending_patch = str(int(patch) + 1)
            version = f"{major}.{minor}.{pending_patch}"
            verbose(f"Bumping patch version to {pending_patch}, change is {version_type}")
        elif version_type == "MINOR":
            pending_minor = str(int(minor) + 1)
            version = f"{major}.{pending_minor}.0"
            verbose(f"Bumping minor version to {pending_minor}, change is {version_type}")
        elif version_type == "MAJOR":  # 6
            # strip off the 'v' add 1 and add the v back in.
            temp = major[1:]  # make sure to hand 1+ digits.
            pending_major = "v" + str(int(temp) + 1)
            version = f"{pending_major}.0.0"
            verbose(f"Bumping major version to {pending_major}, change is {version_type}")

        return version
