# Data Assessment Metrics Library (DAML)

## Description
DAML provides a simple interface to characterize image data and its impact on model performance across classification and object-detection tasks

## Installation
### Dependencies
- python 3.8-3.10
- alibi-detect[tensorflow]
- tensorflow

### Development Environment
#### Ubuntu with Windows Subsystem for Linux
##### Enable Virtual Machine Platform
```
(admin) PS> dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

##### Install WSL2 and Ubuntu (or Linux flavor of choice)
```
(admin) PS> wsl --install
```

##### In Ubuntu, set up your user account and environment
```
:~$ sudo apt update
:~$ sudo apt upgrade -y
:~$ sudo apt install python-is-python3 python3-pip python3-virtualenv -y
```

##### Create the SSH key in your host environment
```
:~$ ssh-keygen -t ed25519 -C "user@domain.com"
```

##### Upload the generated public key (defaults to ~/.ssh/id_ed25519.pub) to GitLab [here](https://gitlab.jatic.net/-/profile/keys).
Additional information on configuring the SSH key can be found [here](https://gitlab.jatic.net/help/user/ssh.md).

##### Clone the daml project from GitLab
```
:~$ git clone git@gitlab.jatic.net:jatic/aria/daml.git
:~$ cd daml
```

#### [Option 1] Working locally with [VirtualEnv](https://virtualenv.pypa.io/en/latest/)

##### Enable the Virtual Environment and install Poetry (packaging) and Tox (test automation)
```
:~/daml$ virtualenv .venv
:~/daml$ source .venv/bin/activate
(.venv) :~/daml$ pip install poetry tox
```

#### [Option 2] VS Code Development Container
This option allows you to run or test DAML in a virtual development container fully isolated from the host environment.  It also allows you to run on different versions of Python independently of what is on your host environment.

_Note: In VS Code, press_ `F1` _or_ `Ctrl+Shift+P` _to open the_ `Command Palette`

1. Open the DAML project in VS Code
2. Install the Remote Development Extension Pack: `ms-vscode-remote.vscode-remote-extensionpack`
3. Using the `Command Palette` run `>Dev Containers: Rebuild and Reopen in Container`
   - On first installation, the container takes a few minutes the prepare the development environment
4. Using the `Command Palette` run `>Python: Select Interpreter`
   - Only select Python versions in the _Workspace_ group, **not** the _Global_ group

The devcontainer is configured to share the SSH keys on the host environment to allow git commands to work.  If you are unable to pull or commit, check the `.ssh` folder in the `$HOME` or `%USERPROFILE%` path and ensure that it is correctly configured.

### Run Tests

DAML uses tox to manage test environments and execution. you can run the tests in several ways.


| Function | Command |
| ------ | ------ |
| Run all tests sequentially | `tox r` |
| Run all tests in parallel | `tox p` |
| Run unit tests sequentially | `tox r -e py38,py39,py310 -- test` |
| Run unit tests in parallel | `tox p -e py38,py39,py310 -- test` |
| Run typecheck sequentially | `tox r -e py38,py39,py310 -- typecheck` |
| Run typecheck in parallel | `tox p -e py38,py39,py310 -- typecheck` |
| Run lint | `tox r -e lint` |


### Install DAML
```
(.venv) :~/daml$ pip install .
```

## POCs
**POC**: Scott Swan @scott.swan

**DPOC**: Andrew Weng @aweng
