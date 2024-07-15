from typing import Any, Dict, List, Optional

from requests import delete, get
from rest import RestWrapper

DAML_PROJECT_URL = "https://harbor.jatic.net/api/v2.0/projects/daml/"
DAML_HARBOR_TOKEN = "DAML_HARBOR_API_TOKEN"

REPOSITORIES = "repositories"
ARTIFACTS = "artifacts"
TAGS = "tags"


class Harbor(RestWrapper):
    """
    Helper class wrapping Harbor REST API calls
    """

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 10,
        verbose: bool = False,
    ):
        super().__init__(DAML_PROJECT_URL, DAML_HARBOR_TOKEN, token, timeout, verbose)
        self.headers = {"Authorization": f"Basic {self.token}"}

    def list_artifacts(self, repository_name: str, tag_filter: Optional[str]) -> List[Dict[str, Any]]:
        """
        List artifacts

        Returns
        -------
        List[Dict[str, Any]]
            List of artifacts

        Notes
        --------
        https://harbor.jatic.net/#/artifact/listArtifacts
        """
        params = {"with_tag": "true", "page_size": "20"}
        if tag_filter:
            params.update({"q": f"tags=~{tag_filter}"})

        r = self._request(get, [REPOSITORIES, repository_name, ARTIFACTS], params)
        return r

    def delete_tag(self, repository_name: str, tag_name: str):
        """
        Delete a tag 'repository_name:tag_name'

        Parameters
        ----------
        repository_name : str
            The name of the repository (i.e. 'cache')
        tag_name : str
            The name of the tag (i.e. 'branch-base-3.11')

        Returns
        -------
        None

        Notes
        --------
        https://harbor.jatic.net/#/artifact/deleteTag
        """

        try:
            self._request(delete, [REPOSITORIES, repository_name, ARTIFACTS, tag_name, TAGS, tag_name])
        except ConnectionError as e:
            status_code = int(str(e))
            # Don't fail if the tag doesn't exist (i.e. function is idempotent)
            if status_code != 404:
                raise e
