# Data Assessment Metrics Library (DAML)

## Description
DAML provides a simple interface to characterize image data and its impact on model performance across classification and object-detection tasks

## Installation
### Environment Dependencies
- [git-lfs](https://git-lfs.com/)

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
**Enable GPG in the host Windows environment**
1. Download and install [GPG4Win](https://www.gpg4win.org/)
2. Create an RSA+RSA 4096 bit GPG Key
   1. `PS> gpg --full-gen-key`
   2. Select RSA and RSA and use 4096-bit key length
   3. Select a validity period
   4. Enter name and email address **matching** your GitLab account
   5. Enter optional comment which displays in parentheses after your name
   6. Confirm your entries
   7. Set a strong password
3. Note down the `<KEY ID>`
   1. `PS> gpg --list-secret-keys --keyid-format LONG`
      * The key begins after the encryption method:<br>
      `sec   rsa4096/<KEY ID>`
4. Export the public key to upload to GitLab
   1. `PS> gpg --armor --export <KEY ID>`

**Forward GPG pin entry requests from WSL to Windows**
1. Ensure gpg is installed
   - `$: sudo apt install gpg`
2. Register pin entry client
   - `$: echo pinentry-program /mnt/c/Program\ Files\ \(x86\)/Gpg4win/bin/pinentry.exe > ~/.gnupg/gpg-agent.conf`
3. Reload the gpg agent
   - `$: gpg-connect-agent reloadagent /bye`

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

DAML uses containers to normalize environments for testing and development.  The build script automates building docker containers and running tests in parallel.  The run script allows you to run the same commands locally in your devcontainer.

| Function | Container | Local (Devcontainer) |
| -------- | --------- | -------------------- |
| Run unit tests, typecheck and linting GPU | `./build --gpu` | N/A |
| Run unit tests, typecheck and linting | `./build` | N/A |
| Run unit tests on all versions | `./build unit` | N/A |
| Run unit tests and typecheck on python 3.11 | `./build 3.11` | N/A |
| Run only typecheck on python 3.10 | `./build type 3.10` | `pyenv shell daml-3.10; ./run type` |
| Run only unit tests on python 3.8 | `./build unit 3.8` | `pyenv shell daml-3.8; ./run unit` |
| Run only unit tests on python 3.9 w/ GPU enabled | `./build unit 3.9 --gpu` | `pyenv shell daml-3.9; ./run unit 3.9` |
| Build documentation | `./build docs` | `pyenv shell daml-3.11; ./run docs` |

- Note: The python version argument is optional for `./run`, and it will use the active version of python if not specified.
- Note: The `./run` command executes on your local devcontainer which already has GPU access enabled.
- Note: Adding a convenience parameter to `./run` is coming soon.
