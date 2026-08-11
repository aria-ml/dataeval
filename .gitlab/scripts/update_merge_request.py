#!/usr/bin/env python3
import sys

GITLAB_URL = "https://gitlab.jatic.net/jatic/aria/dataeval/-"
PAGES_URL = "https://jatic.pages.jatic.net/aria/dataeval"

if __name__ == "__main__":
    from gitlab import Gitlab

    merge_request_iid = int(sys.argv[1])
    pipeline_iid = int(sys.argv[2])

    gl = Gitlab(verbose=True)
    pipeline_jobs = {job["name"]: job for job in gl.get_pipeline_jobs(pipeline_iid)}

    pipeline_url = f"{GITLAB_URL}/pipelines/{pipeline_iid}"
    pipeline_link = f"[{pipeline_iid}]({pipeline_url})"

    # Preview site published by the 'pages preview' job as a parallel Pages
    # deployment under the mr-<iid> path prefix.
    preview_url = f"{PAGES_URL}/mr-{merge_request_iid}"

    coverage_pct = pipeline_jobs["coverage"]["coverage"]
    coverage_link = f"[coverage ({coverage_pct}%)]({preview_url}/coverage/)"

    note = f"Pipeline {pipeline_link} done - review {coverage_link}"

    if "docs" in pipeline_jobs:
        docs_link = f"[documentation]({preview_url}/docs/)"
        note = f"{note} and {docs_link}"

    print("Updating merge request with job results...")
    gl.create_merge_request_note(merge_request_iid, note)
