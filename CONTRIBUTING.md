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

##### Create the GPG key in your host environment
GitLab provides documentation for the GPG signing process [here](https://docs.gitlab.com/ee/user/project/repository/signed_commits/gpg.html).

###### Windows/WSL/Dev Containers
####### Windows
1. Download and install [GPG4Win](https://www.gpg4win.org/)
2. Create an RSA+RSA 4096 bit GPG Key (Windows Instructions Below):
   1. Open Kleopatra (GPG4Win UI)
   2. `File` > `New OpenPGP Key Pair... (Ctrl+N)`
   3. Enter name and email address **matching** your GitLab account
   4. Check `Protect the generated key with a passphrase`
   5. Click `Advanced Settings`
   6. Set `Key Material` to `RSA + RSA @ 4096 bits` [source](https://docs.gitlab.com/ee/user/project/repository/signed_commits/gpg.html#create-a-gpg-key)
   7. Check `Signing` and `Authentication` and uncheck `Valid until: expiry date`.
   8. Click OK, set your passphrase, and your key should be visible in the list.

####### WSL
1. Ensure gpg is installed: `sudo apt install gpg`
2. Register pinentry: `echo pinentry-program /mnt/c/Program\ Files\ \(x86\)/Gpg4win/bin/pinentry.exe > ~/.gnupg/gpg-agent.conf`
3. Reload the gpg: `gpg-connect-agent reloadagent /bye`

###### Linux
Follow the GitLab instructions provided in the section heading.  The instructions can also be run directly in WSL2 but is not as straightforward to link to a running Dev Container as following the instructions for Windows/WSL2 + Dev Containers

##### Clone the daml project from GitLab
```
:~$ git clone git@gitlab.jatic.net:jatic/aria/daml.git
:~$ cd daml
```

##### Configure .gitconfig user settings
```
:~$ git config --global user.name "User Name"
:~$ git config --global user.email "username@domain.com"
```

##### Configure git to sign commits
```
:~$ git config --global user.signingkey <KEY ID>
:~$ git config --global commit.gpgsign true
```

##### Configure .gitconfig safe directory settings
This is required for git to trust your repository ownership when the files belong to a different user identity, which is possible when running under different virtual environments/containers.  Do not set this if you are using a shared device, repository or file system unless you know what you're doing.
```
:~$ git config --global --add safe.directory "*"
```

#### VS Code Development Container
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
| ~~Run all tests in parallel~~ | ~~`tox p`~~ |
| Run unit tests sequentially | `tox r -e py38,py39,py310 -- test` |
| ~~Run unit tests in parallel~~ | ~~`tox p -e py38,py39,py310 -- test`~~ |
| Run typecheck sequentially | `tox r -e py38,py39,py310 -- typecheck` |
| Run typecheck in parallel~~ | ~~`tox p -e py38,py39,py310 -- typecheck` |
| Run unit tests for specific python version | `tox r -e py3* -- test` |
| Run typecheck for specific python version | `tox r -e py3* -- typecheck` |
| Run lint | `tox r -e lint` |

As GPU access is not thread-safe, tests cannot be run in parallel.
