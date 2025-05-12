from os import environ, path
from typing import Any, Callable, Dict, Sequence, Union, cast

from requests import JSONDecodeError, Response


class _VerboseSingleton:
    verbose = False

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance


def verbose(text: str) -> None:
    if _VerboseSingleton().verbose:
        print(text)


def set_verbose(value: bool) -> None:
    _VerboseSingleton().verbose = value


def replace_long_strings(d, max_length, replacement=None):
    if isinstance(d, dict):
        return {k: replace_long_strings(v, max_length, replacement) for k, v in d.items()}
    if isinstance(d, list):
        return [replace_long_strings(i, max_length, replacement) for i in d]
    if isinstance(d, str) and len(d) > max_length:
        return replacement or f"{d[: max_length - 3]}..."
    return d


class RestWrapper:
    """
    Helper class wrapping generic REST API calls
    """

    def __init__(
        self,
        project_url: str,
        env_token: str,
        override_token: str | None = None,
        timeout: int = 10,
        verbose: bool = False,
    ) -> None:
        self.headers: dict
        self.project_url = project_url
        self.token = environ[env_token] if override_token is None else override_token
        if self.token is None or len(self.token) == 0:
            raise ValueError("No token provided")
        self.timeout = timeout
        set_verbose(verbose)

    def _get_param_str(self, params: Dict[str, Any] | None) -> str:
        if params is None:
            return ""
        return "&".join({f"{k}={v}" for k, v in params.items()})

    def _request(
        self,
        fncall: Callable,
        resource: Union[str, Sequence[str]],
        params: Dict[str, Any] | None = None,
        data: Dict[str, Any] | None = None,
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
        params : Dict[str, Any] | None, default None
            Optional parameters for the resource
        data : Dict[str, Any] | None, default None
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
        while page <= last_page:
            params_str = self._get_param_str(params)
            url_with_params = f"{url}?{params_str}" if params_str else url
            args = {
                "url": url_with_params,
                "headers": self.headers,
                dtype: data,
                "timeout": self.timeout,
            }

            response = cast(Response, fncall(**args))

            args_to_print = replace_long_strings({x: args[x] for x in args if x != "headers"}, 50)
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
            elif response.links and len(response.links) > 0:
                last_page = len(response.links) + 1
            else:
                if not isinstance(response_json, (dict, list)):
                    raise TypeError("Response JSON type is not a dict or list")
                return response_json

            if not isinstance(response_json, list):
                raise TypeError("Response JSON type is not a list")
            page += 1
            params["page"] = str(page)
            result += response_json

        return result
