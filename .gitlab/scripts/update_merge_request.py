#!/usr/bin/env python3
import sys

GITLAB_URL = "https://gitlab.jatic.net/jatic/aria/daml/-"
PAGES_URL = "https://jatic.pages.jatic.net/-/aria/daml/-"

if __name__ == "__main__":
    from gitlab import Gitlab

    merge_request_iid = int(sys.argv[1])
    pipeline_iid = int(sys.argv[2])

    gl = Gitlab(verbose=True)
    pipeline_jobs = {job["name"]: job for job in gl.get_pipeline_jobs(pipeline_iid)}

    pipeline_url = f"{GITLAB_URL}/pipelines/{pipeline_iid}"
    pipeline_link = f"[{pipeline_iid}]({pipeline_url})"

    coverage_pct = pipeline_jobs["coverage"]["coverage"]
    coverage_job_id = pipeline_jobs["coverage"]["id"]
    coverage_url = f"{PAGES_URL}/jobs/{coverage_job_id}/artifacts/htmlcov/index.html"
    coverage_link = f"[coverage ({coverage_pct}%)]({coverage_url})"

    note = f"Pipeline {pipeline_link} done - review {coverage_link}"

    if "docs" in pipeline_jobs:
        docs_job_id = pipeline_jobs["docs"]["id"]
        docs_url = f"{PAGES_URL}/jobs/{docs_job_id}/artifacts/output/docs/html/index.html"
        docs_link = f"[documentation]({docs_url})"
        note = f"{note} and {docs_link}"

    print("Updating merge request with job results...")
    gl.create_merge_request_note(merge_request_iid, note)
