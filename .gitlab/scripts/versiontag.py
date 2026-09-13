import re
from typing import Literal

from gitlab import Gitlab
from rest import verbose

# Prerelease kinds, in PEP 440 precedence order: v1.0.0-a0 < v1.0.0-rc0 < v1.0.0.
# 'a' is an early snapshot of main published so downstream projects can build
# against it; 'rc' is a candidate for the release that follows.
PRERELEASE_KINDS = ("a", "rc")
# Regex alternation for the kinds above, shared with releasegen.py
PRERELEASE_KINDS_RE = "|".join(PRERELEASE_KINDS)

# Pattern to match prerelease versions like v1.0.0-a0 or v1.0.0-rc0
PRERELEASE_PATTERN = re.compile(rf"v([0-9]+)\.([0-9]+)\.([0-9]+)-({PRERELEASE_KINDS_RE})([0-9]+)$")
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
        match = PRERELEASE_PATTERN.match(self.current)
        return f"v{match[1]}.{match[2]}.{match[3]}" if match else self.current

    @property
    def prerelease_kind(self) -> str | None:
        """The kind ('a' or 'rc') of the current prerelease, or None for a full release."""
        match = PRERELEASE_PATTERN.match(self.current)
        return match[4] if match else None

    @property
    def is_prerelease(self) -> bool:
        """Returns True if the current version is a prerelease."""
        return self.prerelease_kind is not None

    def _next_base(self, version_type: Literal["MAJOR", "MINOR", "PATCH"]) -> str:
        """Bump the current base version by the requested type, ignoring any prerelease suffix."""
        major, minor, patch = self.current_base.split(".")
        if version_type == "PATCH":
            return f"{major}.{minor}.{int(patch) + 1}"
        if version_type == "MINOR":
            return f"{major}.{int(minor) + 1}.0"
        if version_type == "MAJOR":
            # strip off the 'v' add 1 and add the v back in - make sure to handle 1+ digits
            return f"v{int(major[1:]) + 1}.0.0"
        raise ValueError(f"Unknown version type {version_type!r}.")

    def next(self, version_type: Literal["MAJOR", "MINOR", "PATCH"]) -> str:
        # If current is a prerelease, finalize it by stripping the suffix
        if self.is_prerelease:
            version = self.current_base
            verbose(f"Finalizing prerelease {self.current} to {version}")
            return version

        version = self._next_base(version_type)
        verbose(f"Bumping version from {self.current} to {version}, change is {version_type}")
        return version

    def next_prerelease(
        self,
        version_type: Literal["MAJOR", "MINOR", "PATCH"],
        kind: Literal["a", "rc"] = "rc",
    ) -> str:
        """
        Calculate the next prerelease version of the requested kind.

        Same kind as the current prerelease (v1.0.0-a0, 'a') increments the sequence
        number (v1.0.0-a1). Promoting up the ladder (v1.0.0-a3, 'rc') restarts at 0 on
        the same base version (v1.0.0-rc0). From a full release, the base version is
        bumped per `version_type` and the sequence starts at 0.
        """
        if kind not in PRERELEASE_KINDS:
            raise ValueError(f"Unknown prerelease kind {kind!r}; expected one of {PRERELEASE_KINDS}.")

        current = self.current
        current_kind = self.prerelease_kind

        if current_kind is not None:
            if current_kind == kind:
                base, _, number = current.rpartition(f"-{kind}")
                version = f"{base}-{kind}{int(number) + 1}"
                verbose(f"Incrementing prerelease from {current} to {version}")
                return version

            if PRERELEASE_KINDS.index(kind) > PRERELEASE_KINDS.index(current_kind):
                version = f"{self.current_base}-{kind}0"
                verbose(f"Promoting prerelease from {current} to {version}")
                return version

            # Going back down the ladder would publish a version that sorts before one
            # already on PyPI, so refuse rather than emit an out-of-order release.
            raise ValueError(
                f"Cannot create a '{kind}' prerelease after {current}: "
                f"{self.current_base}-{kind}0 sorts before the already published {current}."
            )

        version = f"{self._next_base(version_type)}-{kind}0"
        verbose(f"Creating new prerelease {version} from {current}, change is {version_type}")
        return version
