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

##### Known Issues
In some cases, the pip cache can get corrupted and will need to be cleaned. If you see errors like `FAIL code -9` or `Killed` during tox execution of a pip install, the following workaround may help:
###### Linux
```
:~/daml$ rm -rf ~/.cache/pip
```

###### Windows
```
PS> Remove-Item -Recurse -Force %LOCALAPPDATA%/pip/cache
```

**Note:** this will clear out your entire pip cache so future installations will need to download new copies of any packages

##### Sharing your SSH key with Development Container
###### Windows:
Start a local Administrator PowerShell and run the following commands:

```
(admin) PS> Set-Service ssh-agent -StartupType Automatic
(admin) PS> Start-Service ssh-agent
(admin) PS> Get-Service ssh-agent
```

###### Linux:
First, start the SSH Agent in the background by running the following in a terminal:

```
:~$ eval "$(ssh-agent -s)"
```

Then add these lines to your ~/.bash_profile or ~/.zprofile (for Zsh) so it starts on login:

```
if [ -z "$SSH_AUTH_SOCK" ]; then
   # Check for a currently running instance of the agent
   RUNNING_AGENT="`ps -ax | grep 'ssh-agent -s' | grep -v grep | wc -l | tr -d '[:space:]'`"
   if [ "$RUNNING_AGENT" = "0" ]; then
        # Launch a new instance of the agent
        ssh-agent -s &> $HOME/.ssh/ssh-agent
   fi
   eval `cat $HOME/.ssh/ssh-agent`
fi
```

#### Run Tests

##### Run all tests
```
(.venv) :~/daml$ tox
```

##### Run selective test
```
(.venv) :~/daml$ tox -e [lint, test-{py38,py39,py310}, typecheck-{py38,py39,py310}]
```

#### Install DAML
```
(.venv) :~/daml$ pip install .
```

## POCs
**POC**: Scott Swan @scott.swan

**DPOC**: Andrew Weng @aweng
