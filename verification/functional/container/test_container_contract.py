"""Verify the container build definitions hold to the contract the images claim.

These are static checks over the repository, so they run in the normal
verification lane without Docker. They cover the properties that are asserted
in documentation and in the program-standards record but that nothing else
would catch if a Dockerfile were edited by hand:

  - a Dockerfile exists for every declared variant           [CS-1-H-2]
  - the generated Dockerfiles match their template           (stale-output gate)
  - the image runs as a non-root user                        [CS-2-H-1]
  - the base image comes from a trusted registry             [CS-1-S-2]
  - no ENTRYPOINT, and CMD explicitly cleared                [IR-2.3, IR-2.4]
  - the published stage carries a scan report and an SBOM    [CS-2-H-3, CS-2-H-4]
  - variants.yaml extras agree with the exported requirements

The runtime behavior of a built image is covered separately by
``docker/smoke.py``, which the Dockerfile's test stage runs during the build.
"""

from __future__ import annotations

import os
import re
from importlib.metadata import metadata
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = PROJECT_ROOT / "docker"
VARIANTS_FILE = DOCKER_DIR / "variants.yaml"

# CS-1-S-2 names two trusted sources: Iron Bank and Docker Hub Official Images.
#
# Official Images are single-segment repository names with no namespace, which
# is what distinguishes `ubuntu:24.04` from a vendor image such as
# `nvidia/cuda:...`. Iron Bank images are served from registry1.dso.mil.
# Anything else -- a vendor namespace on Docker Hub, ghcr.io, quay.io -- is not
# a trusted source under this requirement.
OFFICIAL_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*:[a-zA-Z0-9][a-zA-Z0-9._-]*$")
IRON_BANK_RE = re.compile(r"^registry1\.dso\.mil/ironbank/")


def _is_trusted_base(image: str) -> bool:
    return bool(OFFICIAL_IMAGE_RE.match(image) or IRON_BANK_RE.match(image))


def _variants() -> dict:
    return yaml.safe_load(VARIANTS_FILE.read_text())


VARIANT_NAMES = sorted(_variants()["variants"])


@pytest.fixture(scope="module")
def config() -> dict:
    return _variants()


@pytest.mark.parametrize("variant", VARIANT_NAMES)
class TestDockerfiles:
    """Each declared variant has a Dockerfile that meets the container standards."""

    def test_dockerfile_exists(self, variant: str):
        """CS-1-H-2: the image is defined by a Dockerfile in the source repository."""
        assert (DOCKER_DIR / f"Dockerfile.{variant}").is_file(), f"missing docker/Dockerfile.{variant}"

    def test_declares_non_root_user(self, variant: str):
        """CS-2-H-1: the final stage switches to a non-root user."""
        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        users = re.findall(r"^USER\s+(\S+)", text, re.MULTILINE)
        assert users, "no USER instruction; the image would run as root"
        assert users[-1] not in {"root", "0"}, f"final USER is {users[-1]!r}"

    def test_base_image_is_from_a_trusted_registry(self, variant: str, config: dict):
        """CS-1-S-2: base images come from Iron Bank or Docker Hub Official Images."""
        base = config["variants"][variant]["base_image"]
        assert _is_trusted_base(base), (
            f"base_image {base!r} is neither a Docker Hub Official Image nor an Iron Bank image. "
            "Vendor-namespaced images (nvidia/cuda, ...) are not a trusted source under CS-1-S-2."
        )

    def test_declares_no_entrypoint(self, variant: str):
        """No ENTRYPOINT: the image is an environment, not a CLI tool or service.

        The hard requirements in IR-2.3 and IR-2.4 are each scoped to CLI tools
        or to services. An image that declares neither is neither, which is what
        keeps those requirements out of scope for this artifact.
        """
        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        assert not re.search(r"^ENTRYPOINT", text, re.MULTILINE), (
            "an ENTRYPOINT makes this image arguable as a CLI tool, which pulls "
            "IR-2.4-H-1 (--help output) back into scope"
        )

    def test_clears_inherited_cmd(self, variant: str):
        """CMD is explicitly cleared, not merely omitted.

        ubuntu:24.04 sets ``CMD ["/bin/bash"]``. Omitting CMD inherits it, and
        the published image would then advertise a command this project never
        chose -- visible to anyone running ``docker image inspect``.
        """
        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        cmds = re.findall(r"^CMD\s+(.*)$", text, re.MULTILINE)
        assert cmds == ["[]"], f"expected exactly one `CMD []`, found {cmds!r}"


class TestVariantConsistency:
    """The variant definitions agree with the rest of the project."""

    def test_variants_match_device_variants(self, config: dict):
        """variants.yaml covers exactly the device variants the noxfile builds for."""
        noxfile = (PROJECT_ROOT / "noxfile.py").read_text()
        match = re.search(r"^DEVICE_VARIANTS\s*=\s*\[(.*?)\]", noxfile, re.MULTILINE | re.DOTALL)
        assert match, "could not find DEVICE_VARIANTS in noxfile.py"
        device_variants = sorted(re.findall(r'"([^"]+)"', match.group(1)))
        assert sorted(config["variants"]) == device_variants

    def test_python_version_is_supported(self, config: dict):
        """The interpreter shipped in the image is one the project supports.

        Read from installed distribution metadata rather than pyproject.toml:
        the verification lane runs on 3.10, where ``tomllib`` is not in the
        standard library, and the built metadata is the more faithful source.
        """
        classifiers = metadata("dataeval").get_all("Classifier") or []
        supported = {
            c.rsplit(" :: ", 1)[-1] for c in classifiers if c.startswith("Programming Language :: Python :: 3.")
        }
        assert supported, "dataeval metadata declares no Python version classifiers"
        assert config["python_version"] in supported, (
            f"image ships Python {config['python_version']}, which is not in {sorted(supported)}"
        )

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_extras_match_exported_requirements(self, variant: str, config: dict):
        """The image installs the same extras that requirements.<variant>.txt was exported with.

        ``_export_dependency_files`` in noxfile.py writes each requirements file
        from ``--extra <variant> --extra onnx[-<variant>] --extra opencv``, and
        the dependency scanner reads those files. If the image installed a
        different set, the scanned surface and the shipped surface would differ.
        """
        expected = [variant, "onnx" if variant == "cpu" else f"onnx-{variant}", "opencv"]
        assert config["variants"][variant]["extras"] == expected

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_extras_are_declared(self, variant: str, config: dict):
        """Every extra a variant installs actually exists on the built distribution.

        ``uv sync --extra <name>`` fails loudly on an unknown extra, so this
        catches the typo at verification time rather than partway through a
        container build.
        """
        declared = set(metadata("dataeval").get_all("Provides-Extra") or [])
        assert declared, "dataeval metadata declares no extras"
        missing = set(config["variants"][variant]["extras"]) - declared
        assert not missing, f"variants.yaml references undeclared extras: {sorted(missing)}"


class TestGeneratedFilesAreCurrent:
    """The committed Dockerfiles are what the template currently renders."""

    def test_dockerfiles_are_marked_generated(self):
        """A hand edit should be obviously wrong to whoever opens the file."""
        for variant in VARIANT_NAMES:
            first_line = (DOCKER_DIR / f"Dockerfile.{variant}").read_text().splitlines()[0]
            assert "AUTO-GENERATED" in first_line, f"Dockerfile.{variant} lost its generated-file banner"

    def test_template_and_variants_are_tracked_together(self):
        """Both inputs to generation exist; `nox -s docker_check` proves they are in sync."""
        assert (DOCKER_DIR / "Dockerfile.j2").is_file()
        assert VARIANTS_FILE.is_file()
        assert (DOCKER_DIR / "generate.py").is_file()

    def test_generation_does_not_depend_on_git(self):
        """Rendering must be a pure function of the template and variants.yaml.

        An earlier version derived the ARG default from ``git describe``. CI
        clones shallow and without tags, so it rendered "unknown" -- which is
        not valid PEP 440 and would fail SETUPTOOLS_SCM_PRETEND_VERSION -- and
        ``docker_check`` failed on every pipeline. It also made the output
        differ per branch, so cherry-picks carried a spurious diff.
        """
        source = (DOCKER_DIR / "generate.py").read_text()
        assert "subprocess" not in source, "generate.py shells out again; rendering must not depend on repository state"

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_arg_default_is_a_valid_placeholder_version(self, variant: str):
        """The build-arg default must parse as PEP 440.

        It is fed to SETUPTOOLS_SCM_PRETEND_VERSION by the build stage, so a
        local ``docker build`` without ``--build-arg`` fails outright if this is
        not a version string.
        """
        from packaging.version import InvalidVersion, Version

        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        match = re.search(r'^ARG DATAEVAL_VERSION="([^"]*)"', text, re.MULTILINE)
        assert match, "Dockerfile declares no DATAEVAL_VERSION default"
        default = match.group(1)
        try:
            Version(default)
        except InvalidVersion:
            pytest.fail(f"ARG DATAEVAL_VERSION default {default!r} is not valid PEP 440")


class TestScanReport:
    """CS-2-H-3: the published image carries its own vulnerability scan report.

    The report cannot be produced inside a single build -- it describes the
    finished filesystem -- so CI builds ``prod``, scans it, writes the report
    into ``docker/security/`` and builds ``scanned``, which is ``prod`` plus
    that one layer. These checks pin the parts of that arrangement which live
    in source; ``verify:docker`` checks the report in the published artifact.
    """

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_scanned_stage_extends_prod(self, variant: str):
        """The published stage must add to ``prod`` rather than rebuild it.

        Rebuilding would scan one image and publish another.
        """
        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        assert re.search(r"^FROM prod AS scanned\b", text, re.MULTILINE), (
            "Dockerfile has no `FROM prod AS scanned` stage; the scanned image "
            "must be the scanned filesystem plus the report, not a second build"
        )

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_scanned_stage_copies_the_report(self, variant: str):
        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        scanned = text.split("FROM prod AS scanned", 1)[-1]
        assert re.search(r"^COPY .*docker/security/", scanned, re.MULTILINE), (
            "the scanned stage does not copy docker/security/ into the image"
        )

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_prod_stage_does_not_need_the_report(self, variant: str):
        """``--target prod`` must build without a scan having run.

        The report only exists once ``prod`` has been built and scanned, so a
        reference to it from ``prod`` itself would be a build-order cycle, and
        would break a plain local ``docker build``.
        """
        text = (DOCKER_DIR / f"Dockerfile.{variant}").read_text()
        prod = text.split("AS prod", 1)[-1].split("FROM prod AS scanned", 1)[0]
        # Comments explaining the scanned stage sit above its FROM and so fall
        # inside this slice; only instructions constrain the build.
        prod = "\n".join(ln for ln in prod.splitlines() if not ln.lstrip().startswith("#"))
        assert "docker/security" not in prod, (
            "the prod stage references docker/security; it must not, or `--target prod` "
            "cannot build before the scan that produces it"
        )

    def test_build_context_admits_the_report(self):
        """``.dockerignore`` excludes ``docker/``; the report must be re-included."""
        ignore = (PROJECT_ROOT / ".dockerignore").read_text()
        assert re.search(r"^!docker/security\b", ignore, re.MULTILINE), (
            ".dockerignore excludes docker/ without re-including docker/security, "
            "so the COPY in the scanned stage would fail"
        )

    def test_scanner_is_pinned_at_or_above_the_required_version(self):
        """CS-2-H-2 names Trivy v0.62.0 as the floor for an approved scanner."""
        from packaging.version import Version

        ci = (PROJECT_ROOT / ".gitlab" / "ci" / "container.yml").read_text()
        match = re.search(r'^\s*TRIVY_VERSION:\s*"([^"]+)"', ci, re.MULTILINE)
        assert match, "container.yml does not pin TRIVY_VERSION"
        assert Version(match.group(1)) >= Version("0.62.0"), (
            f"Trivy {match.group(1)} is below the v0.62.0 floor named by CS-2-H-2"
        )

    def test_latest_is_a_retag_not_a_build(self):
        """`latest` must point at a published release, never be built separately.

        A second build could differ from the release it claims to be, and would
        not be covered by that release's signature or scan.
        """
        ci = (PROJECT_ROOT / ".gitlab" / "ci" / "container.yml").read_text()
        promote = ci.split("promote:latest:", 1)[-1].split("\nrelease:sbom:", 1)[0]
        assert "imagetools create" in promote, "promote:latest does not retag; it must not rebuild"
        assert "buildx build" not in promote, "promote:latest builds an image; latest must be a retag"

    def test_latest_is_gated_on_the_scan(self):
        """A release that failed its scan must not take the default pointer."""
        ci = (PROJECT_ROOT / ".gitlab" / "ci" / "container.yml").read_text()
        promote = ci.split("promote:latest:", 1)[-1].split("\nrelease:sbom:", 1)[0]
        assert "container_scanning" in promote, "promote:latest does not depend on container_scanning"

    def test_only_the_highest_release_takes_latest(self):
        """A patch on an older line must not drag `latest` backwards."""
        script = DOCKER_DIR / "promote-latest.sh"
        assert script.is_file(), "docker/promote-latest.sh is missing"
        import subprocess

        def promote(tag: str, tags: str) -> str:
            return subprocess.run(
                [str(script)],
                input=tags,
                capture_output=True,
                text=True,
                env={"PATH": os.environ["PATH"], "CI_COMMIT_TAG": tag},
            ).stdout.strip()

        assert promote("v1.1.3", "v1.1.0\nv1.1.2\nv1.1.3") == "latest"
        assert promote("v1.1.4", "v1.1.0\nv1.2.0\nv1.1.4") == "", "a patch outranked by a newer line took latest"
        assert promote("v1.10.0", "v1.2.0\nv1.9.0\nv1.10.0") == "latest", "version ordering is lexical, not numeric"
        assert promote("v1.2.0-rc1", "v1.1.0\nv1.2.0-rc1") == "", "a prerelease took latest"
        assert promote("", "v1.1.0") == "", "a branch build took latest"

    def test_sbom_is_built_before_the_push(self):
        """CS-2-H-4: the SBOM must describe the image, and ship with it.

        It is generated from the same tarball trivy scanned, so it has to be
        produced before the push rather than from the pushed reference.
        """
        ci = (PROJECT_ROOT / ".gitlab" / "ci" / "container.yml").read_text()
        push = ci.split("push:docker:", 1)[-1].split("\nvalidate:docker:", 1)[0]
        sbom = re.search(r"syft scan docker-archive:[^\n]*docker/security/sbom\.cdx\.json", push)
        assert sbom, "push:docker does not generate the SBOM into docker/security/"
        assert push.index(sbom.group(0)) < push.index("--push"), (
            "the SBOM is generated after the push, so it cannot be copied into the image"
        )

    def test_sbom_is_attested_to_the_registry(self):
        """CS-2-H-4 requires the SBOM stored alongside the image in the registry."""
        ci = (PROJECT_ROOT / ".gitlab" / "ci" / "container.yml").read_text()
        assert re.search(r"cosign attest .*--type cyclonedx", ci), (
            "no cosign CycloneDX attestation; an in-image copy alone does not satisfy CS-2-H-4"
        )

    def test_scan_gates_the_push(self):
        """A HIGH or CRITICAL finding must fail the job before anything is pushed."""
        ci = (PROJECT_ROOT / ".gitlab" / "ci" / "container.yml").read_text()
        push = ci.split("push:docker:", 1)[-1].split("\nvalidate:docker:", 1)[0]
        gate = re.search(r"trivy image[^\n]*--severity HIGH,CRITICAL[^\n]*--exit-code 1", push)
        assert gate, "push:docker has no blocking HIGH/CRITICAL trivy gate"
        assert push.index(gate.group(0)) < push.index("--push"), (
            "the trivy gate runs after the push; a failing scan would not withhold the image"
        )
