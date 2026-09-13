import re
from typing import Literal

from gitlab import Gitlab
from rest import verbose

# Pattern to match prerelease versions like v1.0.0-rc0
PRERELEASE_PATTERN = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)-rc([0-9]+)")
# Pattern to match standard versions like v1.0.0
VERSION_PATTERN = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)$")


class VersionTag:
    """Provides current and pending/next version number for DataEval."""

    def __init__(self, gitlab: Gitlab) -> None:
        self.gl = gitlab
        self._current = None
        self._pending = None

    @property
    def current(self) -> str:
        """
        The current version of DataEval retrieved from repository tags.
        Matches both standard versions (v1.0.0) and prereleases (v1.0.0-rc0).
        """
        if self._current is None:
            tags = self.gl.list_tags()
            for tag in tags:
                name = tag["name"]
                # Accept both standard versions and prerelease versions
                if VERSION_PATTERN.match(name) or PRERELEASE_PATTERN.match(name):
                    self._current = name
                    break
            if self._current is None:
                raise ValueError("Unable to get current version.")
        return self._current

    @property
    def current_base(self) -> str:
        """
        The current base version (without prerelease suffix).
        For v1.0.0-rc0 returns v1.0.0, for v1.0.0 returns v1.0.0.
        """
        current = self.current
        if "-rc" in current:
            return current.split("-rc")[0]
        return current

    @property
    def is_prerelease(self) -> bool:
        """Returns True if the current version is a prerelease."""
        return "-rc" in self.current

    def next(self, version_type: Literal["MAJOR", "MINOR", "PATCH"]):
        current = self.current

        # If current is a prerelease, finalize it by stripping the -rcX suffix
        if self.is_prerelease:
            version = self.current_base
            verbose(f"Finalizing prerelease {current} to {version}")
            return version

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

    def next_prerelease(self, version_type: Literal["MAJOR", "MINOR", "PATCH"]) -> str:
        """
        Calculate next prerelease version.

        If current is already a prerelease (v1.0.0-rc0), increment rc number (v1.0.0-rc1).
        Otherwise, calculate new base version and start at rc0.
        """
        current = self.current

        # If current is already a prerelease, increment rc number
        if self.is_prerelease:
            base, rc_part = current.split("-rc")
            next_rc = int(rc_part) + 1
            version = f"{base}-rc{next_rc}"
            verbose(f"Incrementing prerelease from {current} to {version}")
            return version

        # Otherwise, calculate new base version and start at rc0
        # Use the base version calculation but don't finalize
        version = current
        major, minor, patch = current.split(".")
        if version_type == "PATCH":
            pending_patch = str(int(patch) + 1)
            version = f"{major}.{minor}.{pending_patch}"
        elif version_type == "MINOR":
            pending_minor = str(int(minor) + 1)
            version = f"{major}.{pending_minor}.0"
        elif version_type == "MAJOR":
            temp = major[1:]
            pending_major = "v" + str(int(temp) + 1)
            version = f"{pending_major}.0.0"

        prerelease_version = f"{version}-rc0"
        verbose(f"Creating new prerelease {prerelease_version} from {current}, change is {version_type}")
        return prerelease_version
