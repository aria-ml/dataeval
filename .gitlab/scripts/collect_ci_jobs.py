#!/usr/bin/env python3
"""Record the result of each job in this pipeline as ``output/ci_jobs.json``.

The verification registry cites pipeline jobs as evidence (``CI: <job>``). ``generate_metarepo.py`` reads this file
to turn those citations into Pass/Fail. Matrix legs are folded into one entry named for the job (``test: [3.13]``
becomes ``test``): it succeeds only when every leg did. Run it from a job that needs the jobs it reports on.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

OUTPUT = Path("output/ci_jobs.json")
OWN_JOB = os.environ.get("CI_JOB_NAME", "")


def job_key(name: str) -> str:
    """Job name without its matrix suffix."""
    return re.sub(r":\s*\[[^\]]*\]$", "", name).strip()


def aggregate(jobs: list[dict]) -> dict[str, str]:
    """Fold job records into ``{job: success | failed | skipped}``; jobs that did not finish are left out."""
    by_key: dict[str, list[str]] = {}
    for job in jobs:
        by_key.setdefault(job_key(job["name"]), []).append(job["status"])
    result = {}
    for key, statuses in by_key.items():
        if "failed" in statuses:
            result[key] = "failed"
        elif all(s == "success" for s in statuses):
            result[key] = "success"
        elif all(s in ("skipped", "manual") for s in statuses):
            result[key] = "skipped"
    return result


def fetch_jobs() -> list[dict]:
    """All jobs of the current pipeline, via the API with the job token."""
    api, project, pipeline = (os.environ[v] for v in ("CI_API_V4_URL", "CI_PROJECT_ID", "CI_PIPELINE_ID"))
    base = f"{api}/projects/{project}/pipelines/{pipeline}/jobs"
    jobs: list[dict] = []
    page = "1"
    while page:
        request = urllib.request.Request(f"{base}?per_page=100&page={page}")
        request.add_header("JOB-TOKEN", os.environ["CI_JOB_TOKEN"])
        with urllib.request.urlopen(request, timeout=30) as response:
            jobs += json.load(response)
            page = response.headers.get("X-Next-Page", "")
    return [job for job in jobs if job["name"] != OWN_JOB]


def main() -> None:
    """Write the file; leave it absent on any failure so the evidence reports Pending instead of guessing."""
    try:
        results = aggregate(fetch_jobs())
    except Exception as error:
        print(f"Could not read the pipeline's jobs ({error}); CI evidence will report as pending", file=sys.stderr)
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Recorded {len(results)} job result(s) in {OUTPUT}:")
    for key, status in sorted(results.items()):
        print(f"  {status:<8}{key}")


if __name__ == "__main__":
    main()
