from os import remove
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
from zipfile import ZipFile

from requests import delete, get, post, put
from rest import RestWrapper

DAML_PROJECT_URL = "https://gitlab.jatic.net/api/v4/projects/151/"
DAML_BUILD_PAT = "DAML_BUILD_PAT"

COMMITS = "repository/commits"
FILES = "repository/files"
MERGE_REQUESTS = "merge_requests"
TAGS = "repository/tags"
BRANCHES = "repository/branches"
PIPELINES = "pipelines"
JOBS = "jobs"
ARTIFACTS = "artifacts"
DOWNLOAD = "download"
PIPELINES = "pipelines"
NOTES = "notes"

LATEST_KNOWN_GOOD = "latest-known-good"


class Gitlab(RestWrapper):
    """
    Helper class wrapping Gitlab REST API calls
    """

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 10,
        verbose: bool = False,
    ):
        super().__init__(DAML_PROJECT_URL, DAML_BUILD_PAT, token, timeout, verbose)
        self.headers = {"PRIVATE-TOKEN": self.token}

    def list_tags(self) -> List[Dict[str, Any]]:
        """
        List project tags

        Returns
        -------
        List[Dict[str, Any]]
            List of project tags

        Notes
        --------
        https://docs.gitlab.com/ee/api/tags.html#list-project-repository-tags
        """
        r = self._request(get, TAGS)
        return r

    def add_tag(self, tag_name: str, ref: str = "main", message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new tag

        Parameters
        ----------
        tag_name : str
            The name of the tag (e.g. "v0.1.0")
        ref : str
            The tag, branch name or SHA to create the tag at
        message : Optional[str], default None
            Create an annotated tag with provided message

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        --------
        https://docs.gitlab.com/ee/api/tags.html#create-a-new-tag
        """
        tag_content = {"tag_name": tag_name, "ref": ref}
        if message is not None:
            tag_content.update({"message": message})
        r = self._request(post, TAGS, tag_content)
        return r

    def delete_tag(self, tag_name: str):
        """
        Delete a tag

        Parameters
        ----------
        tag_name : str
            The name of the tag (e.g. "v0.1.0")

        Returns
        -------
        None

        Notes
        --------
        https://docs.gitlab.com/ee/api/tags.html#delete-a-tag
        """

        try:
            self._request(delete, f"{TAGS}/{tag_name}")
        except ConnectionError as e:
            status_code = int(str(e))
            # Don't fail if the tag doesn't exist (i.e. function is idempotent)
            if status_code != 404:
                raise e

    def get_single_repository_branch(self, branch: str) -> Dict[str, Any]:
        """
        Get a single repository branch

        Parameters
        ----------
        branch_name : str
            The name of the branch (e.g. "releases/vX.Y")

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        --------
        https://docs.gitlab.com/ee/api/branches.html#get-single-repository-branch
        """
        r = self._request(get, f"{BRANCHES}/{branch}")
        return r

    def create_repository_branch(self, branch: str, ref: str) -> Dict[str, Any]:
        """
        Create a repository branch

        Parameters
        ----------
        branch_name : str
            The name of the branch (e.g. "releases/vX.Y")

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        --------
        https://docs.gitlab.com/ee/api/branches.html#create-repository-branch
        """
        r = self._request(post, BRANCHES, {"branch": branch, "ref": ref})
        return r

    def list_merge_requests(
        self,
        state: Optional[Literal["opened", "closed", "locked", "merged"]] = None,
        target_branch: Optional[str] = None,
        source_branch: Optional[str] = None,
        search_title: Optional[str] = None,
        order_by: Optional[Literal["created_at", "title", "merged_at", "updated_at"]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List merge requests

        Returns
        -------
        List[Dict[str, Any]]
            List of merge requests found using criteria

        Notes
        --------
        https://docs.gitlab.com/ee/api/merge_requests.html#list-merge-requests
        """
        params = {"per_page": "100"}
        if state is not None:
            params.update({"state": state})
        if target_branch is not None:
            params.update({"target_branch": target_branch})
        if source_branch is not None:
            params.update({"source_branch": source_branch})
        if search_title is not None:
            params.update({"search": search_title, "in": "title"})
        if order_by is not None:
            params.update({"order_by": order_by})
        r = self._request(get, MERGE_REQUESTS, params)
        return r

    def create_merge_request(
        self,
        title: str,
        description: str,
        source_branch: str = "develop",
        target_branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Create a merge request

        Parameters
        ----------
        title : str
            The title text of the merge request
        description : str
            The description text for the body of the merge request
        source_branch : str, default "develop"
            The source branch that is being merged from
        target_branch : str, default "main"
            The target branch that is being merged into

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        --------
        https://docs.gitlab.com/ee/api/merge_requests.html#create-mr
        """
        r = self._request(
            post,
            MERGE_REQUESTS,
            None,
            {
                "title": title,
                "description": description,
                "source_branch": source_branch,
                "target_branch": target_branch,
            },
        )
        return r

    def update_merge_request(self, mr_iid: int, title: str, description: str) -> Dict[str, Any]:
        """
        Updates a merge request

        Parameters
        ----------
        mr_iid : int
            The id of the merge request to update
        title : str
            The title text of the merge request
        description : str
            The description text for the body of the merge request

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        --------
        https://docs.gitlab.com/ee/api/merge_requests.html#update-mr
        """
        r = self._request(
            put,
            [MERGE_REQUESTS, str(mr_iid)],
            None,
            {"title": title, "description": description},
        )
        return r

    def get_file(self, filepath: str, dest: str, ref: str = "main"):
        """
        Gets the raw file content from the repository

        Parameters
        ----------
        filepath : str
            The filepath for the repository file to download
        dest : str
            The local filepath to save the contents
        ref : str, default "main"
            The tag, branch name or SHA to retrieve the file from

        Notes
        -----
        https://docs.gitlab.com/ee/api/repository_files.html#get-raw-file-from-repository
        """
        r = self._request(get, [FILES, filepath, "raw"], {"ref": ref})
        with open(dest, "wb") as f:
            f.write(r.content)

    def get_file_info(self, filepath: str, ref: str = "develop") -> Dict[str, Any]:
        """
        Gets the file information from the repository

        Parameters
        ----------
        filepath : str
            The filepath for the repository file info to retrieve
        ref : str, default "develop"
            The tag, branch name or SHA to retrieve the file info from

        Returns
        -------
        Dict[str, Any]
            The file information from the specified repository file

        Notes
        -----
        https://docs.gitlab.com/ee/api/repository_files.html#get-file-from-repository
        """
        r = self._request(get, [FILES, filepath], {"ref": ref})
        return r

    def push_file(self, filepath: str, branch: str, commit_message: str, content: str) -> Dict[str, Any]:
        """
        Update a file in the repository

        Parameters
        ----------
        filepath : str
            The filepath of the file to update
        branch : str
            The branch to update the file in
        commit_message : str
            The commit message when commiting the update
        content : str
            The contents of the file to update

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        -----
        https://docs.gitlab.com/ee/api/repository_files.html#update-existing-file-in-repository
        """
        r = self._request(
            put,
            [FILES, filepath],
            None,
            {
                "branch": branch,
                "commit_message": commit_message,
                "content": content,
            },
        )
        return r

    def cherry_pick(self, sha: str, branch: str = "main") -> Dict[str, Any]:
        """
        Cherry pick a commit

        Parameters
        ----------
        sha : str
            The commit sha to cherry pick
        branch : str, default "main"
            The branch to apply the cherry pick to

        Returns
        -------
        Dict[str, Any]:
            The response received after issuing the request

        Notes
        -----
        https://docs.gitlab.com/ee/api/commits.html#cherry-pick-a-commit
        """
        r = self._request(post, [COMMITS, sha, "cherry_pick"], None, {"branch": branch})
        return r

    def get_artifacts(self, job: str, dest: str, ref: str = "main"):
        """
        Gets the artifacts from the last successful pipeline run for the ref specified

        Parameters
        ----------
        job : str
            The job from which to download artifacts
        dest : str
            The local filepath to save the extracted contents
        ref : str, default "main"
            The tag, branch name or SHA to retrieve the file from

        Notes
        -----
        https://docs.gitlab.com/ee/api/job_artifacts.html#download-the-artifacts-archive
        """
        r = self._request(get, [JOBS, ARTIFACTS, ref, DOWNLOAD], {"job": job})
        temp_file = str(uuid4()) + ".zip"
        with open(temp_file, "wb") as f:
            f.write(r.content)
        Path(dest).mkdir(parents=True, exist_ok=True)
        with ZipFile(temp_file, "r") as z:
            z.extractall(dest)
        remove(temp_file)

    def commit(self, branch: str, commit_message: str, actions: List[Dict[str, str]]):
        """
        Commits the specified actions to the repository at the branch specified

        Parameters
        ----------
        branch : str
            The branch to make the commit at
        commit_message : str
            Commit message for the commit
        actions : List[Dict[str, str]]
            A list of actions to commit to the repository

        Notes
        -----
        https://docs.gitlab.com/ee/api/commits.html#create-a-commit-with-multiple-files-and-actions
        """
        r = self._request(
            post,
            [COMMITS],
            None,
            {
                "branch": branch,
                "commit_message": commit_message,
                "actions": actions,
            },
        )
        return r

    def get_pipeline_jobs(self, pipeline_iid: int):
        r = self._request(get, [PIPELINES, str(pipeline_iid), JOBS])
        return r

    def create_merge_request_note(self, merge_request_iid: int, body: str):
        r = self._request(post, [MERGE_REQUESTS, str(merge_request_iid), NOTES], {"body": body})
        return r

    def run_pipeline(self, pipeline_id: int):
        r = self._request(post, ["pipeline_schedules", str(pipeline_id), "play"])
        return r
