from os import environ, path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union, cast

from requests import Response, delete, get, post, put
from verboselog import verbose

DAML_PROJECT_URL = "https://gitlab.jatic.net/api/v4/projects/151/"

COMMITS = "repository/commits"
FILES = "repository/files"
MERGE_REQUESTS = "merge_requests"
TAGS = "repository/tags"
BRANCHES = "repository/branches"

LATEST_KNOWN_GOOD = "latest-known-good"


class Gitlab:
    """
    Helper class wrapping Gitlab REST API calls
    """

    def __init__(
        self,
        project_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 10,
        verbose: bool = False,
    ):
        self.project_url = DAML_PROJECT_URL if project_url is None else project_url
        # $DAML_BUILD_PAT is only available in production environments
        token = environ["DAML_BUILD_PAT"] if token is None else token
        if token is None or len(token) == 0:
            raise ValueError("No token provided")
        self.headers = {"PRIVATE-TOKEN": token}
        self.timeout = timeout
        self.verbose = verbose

    def _request(
        self,
        fncall: Callable,
        resource: Union[str, Sequence[str]],
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Sends requests to Gitlab REST API

        Calling with the following example:
        _request(put, ["resource1", "subpath1"], {"pkey": "pvalue"}, {"dkey": "dvalue"})

        Will run the approximated curl below:
        curl -X PUT -d {"dkey":"dvalue"} https://.../resource1/subpath1?pkey=pvalue

        Parameters
        ----------
        fncall : Callable
            The requests function to call (get, post, put)
        resource : Union[str, Sequence[str]]
            The path(s) of the Gitlab API to call
        params : Optional[Dict[str, Any]], default None
            Optional parameters for the resource
        data : Optional[Dict[str, Any]], default None
            Optional data provided for the request (used in post or put)

        Raises
        ------
        ConnectionError
            Raises if the response status code is not successful (200-299)

        Notes
        --------
        https://docs.gitlab.com/ee/api/rest/
        """
        if not isinstance(resource, str):
            resource = path.join(*resource)

        url = resource
        if params is not None:
            for k, v in params.items():
                url = (url + "?" + k) if url == resource else (url + "&" + k)
                url += "" if v is None else ("=" + v)

        args = {
            "url": self.project_url + url,
            "headers": self.headers,
            "data": data,
            "timeout": self.timeout,
        }

        response = cast(Response, fncall(**args))

        args_to_print = {x: args[x] for x in args if x != "headers"}
        verbose(f"Request '{fncall.__name__}' issued: {args_to_print}")
        verbose(f"Response received: {response}")
        if response.status_code not in range(200, 299):
            raise ConnectionError(response.status_code)
        return response

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
        return r.json()

    def add_tag(
        self, tag_name: str, ref: str = "main", message: Optional[str] = None
    ) -> Dict[str, Any]:
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
        return r.json()

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
        return r.json()

    def list_merge_requests(
        self,
        state: Optional[Literal["opened", "closed", "locked", "merged"]] = None,
        target_branch: Optional[str] = None,
        source_branch: Optional[str] = None,
        search_title: Optional[str] = None,
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
        params = dict()
        if state is not None:
            params.update({"state": state})
        if target_branch is not None:
            params.update({"target_branch": target_branch})
        if source_branch is not None:
            params.update({"source_branch": source_branch})
        if search_title is not None:
            params.update({"search": search_title, "in": "title"})
        r = self._request(get, MERGE_REQUESTS, params)
        return r.json()

    def create_mr(
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
        return r.json()

    def update_mr(self, mr_iid: int, title: str, description: str) -> Dict[str, Any]:
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
        return r.json()

    def get_file(self, filepath: str, dest: str, ref: str = "develop"):
        """
        Gets the raw file content from the repository

        Parameters
        ----------
        filepath : str
            The filepath for the repository file to download
        dest : str
            The local filepath to save the contents
        ref : str, default "develop"
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
        return r.json()

    def push_file(
        self, filepath: str, branch: str, commit_message: str, content: str
    ) -> Dict[str, Any]:
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
        return r.json()

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
        return r.json()
