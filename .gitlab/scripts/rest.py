from os import environ, path
from typing import Any, Callable, Dict, Optional, Sequence, Union, cast

from requests import JSONDecodeError, Response


class _VerboseSingleton:
    verbose = False

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance


def verbose(text: str):
    if _VerboseSingleton().verbose:
        print(text)


def set_verbose(value: bool):
    _VerboseSingleton().verbose = value


class RestWrapper:
    """
    Helper class wrapping generic REST API calls
    """

    def __init__(
        self,
        project_url: str,
        env_token: str,
        override_token: Optional[str] = None,
        timeout: int = 10,
        verbose: bool = False,
    ):
        self.headers: dict
        self.project_url = project_url
        self.token = environ[env_token] if override_token is None else override_token
        if self.token is None or len(self.token) == 0:
            raise ValueError("No token provided")
        self.timeout = timeout
        set_verbose(verbose)

    def _get_param_str(self, params: Optional[Dict[str, Any]]) -> str:
        if params is None:
            return ""
        return "&".join({f"{k}={v}" for k, v in params.items()})

    def _request(
        self,
        fncall: Callable,
        resource: Union[str, Sequence[str]],
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        raw_data: bool = False,
    ) -> Any:
        """
        Sends requests to REST API endpoint

        Calling with the following example:
        _request(put, ["resource1", "subpath1"], {"pkey": "pvalue"}, {"dkey": "dvalue"})

        Will run the approximated curl below:
        curl -X PUT -d {"dkey":"dvalue"} https://.../resource1/subpath1?pkey=pvalue

        Parameters
        ----------
        fncall : Callable
            The requests function to call (get, post, put)
        resource : Union[str, Sequence[str]]
            The path(s) of the API to call
        params : Optional[Dict[str, Any]], default None
            Optional parameters for the resource
        data : Optional[Dict[str, Any]], default None
            Optional data provided for the request (used in post or put)

        Raises
        ------
        ConnectionError
            Raises if the response status code is not successful (200-299)
        """
        if not isinstance(resource, str):
            resource = path.join(*resource)

        url = path.join(self.project_url, resource)

        if params is None:
            params = {}

        if raw_data:
            dtype = "data"
            self.headers["Content-type"] = "application/octet-stream"
        else:
            dtype = "json"
            self.headers["Content-type"] = "application/json"

        last_page = 1
        page = 1
        result = []
        params_str = self._get_param_str(params)
        url_with_params = f"{url}?{params_str}" if params_str else url
        while page <= last_page:
            args = {
                "url": url_with_params,
                "headers": self.headers,
                dtype: data,
                "timeout": self.timeout,
            }

            response = cast(Response, fncall(**args))

            args_to_print = {x: args[x] for x in args if x != "headers"}
            verbose(f"Request '{fncall.__name__}' issued: {args_to_print}")
            verbose(f"Response received: {response}")
            if response.status_code not in range(200, 299):
                raise ConnectionError(response.status_code)

            try:
                response_json = response.json()
            except JSONDecodeError:
                return response

            if "X-Total-Pages" in response.headers:
                last_page = int(response.headers["X-Total-Pages"])
            else:
                assert isinstance(response_json, (dict, list))
                return response_json

            assert isinstance(response_json, list)
            page += 1
            params["page"] = str(page)
            result += response_json

        return result
