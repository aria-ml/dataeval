#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    from gitlab import Gitlab

    merge_request_iid = int(sys.argv[1])
    pipeline_iid = int(sys.argv[2])

    gl = Gitlab(verbose=True)
    jobs = ["coverage", "docs"]
    pipeline_jobs = {job["name"]: job for job in gl.get_pipeline_jobs(pipeline_iid)}

    coverage_pct = pipeline_jobs["coverage"]["coverage"]
    coverage_job_id = pipeline_jobs["coverage"]["id"]
    docs_job_id = pipeline_jobs["docs"]["id"]

    # fmt: off
    pipeline_url = f"https://gitlab.jatic.net/jatic/aria/daml/-/pipelines/{pipeline_iid}"
    coverage_url = f"https://jatic.pages.jatic.net/-/aria/daml/-/jobs/{coverage_job_id}/artifacts/htmlcov/index.html"
    docs_url = f"https://jatic.pages.jatic.net/-/aria/daml/-/jobs/{docs_job_id}/artifacts/output/docs/html/index.html"
    # fmt: on

    pipeline_link = f"[{pipeline_iid}]({pipeline_url})"
    coverage_link = f"[coverage ({coverage_pct}%)]({coverage_url})"
    docs_link = f"[documentation]({docs_url})"

    note = f"Pipeline {pipeline_link} done - review {coverage_link} and {docs_link}"
    print("Updating merge request with coverage and docs note...")
    gl.create_merge_request_note(merge_request_iid, note)
