# Data Assessment Metrics Library (DAML)

## Description
DAML provides a simple interface to characterize image data and its impact on model performance across classification and object-detection tasks

## Installation
- upload an SSH key to your profile:
`User -> Edit Profile -> SSH Keys`

- for details on how to do this: https://gitlab.jatic.net/help/user/ssh.md

- Clone the repo from the JATIC GitLab to your local workspace
```
jgleeson@cdao:$ git clone git@gitlab.jatic.net:jatic/aria/daml.git daml
Cloning into 'daml'...
[SNIP]
Enter passphrase for key '/home/jgleeson/.ssh/id_ed25519':
remote: Enumerating objects: 20, done.
remote: Counting objects: 100% (6/6), done.
remote: Compressing objects: 100% (6/6), done.
remote: Total 20 (delta 2), reused 0 (delta 0), pack-reused 14
Receiving objects: 100% (20/20), 5.20 KiB | 1.73 MiB/s, done.
Resolving deltas: 100% (5/5), done.
jgleeson@cdao:$
```

- create a new feature branch
```
jgleeson@daml:$ git checkout -b issue3-helloworld
Switched to a new branch 'issue3-helloworld'
jgleeson@daml:$
jgleeson@daml:$
jgleeson@daml:$ git status
On branch issue3-helloworld
nothing to commit, working tree clean
jgleeson@daml:$
```

- the project structure is as follows...
```
jgleeson@daml:$ tree .
.
├── README.md
├── docs
├── src
│   └── daml
│       ├── __init__.py
│       └── helloworld.py
└── tests

4 directories, 3 files
jgleeson@daml:$
```

## Usage
TODO

## Remote Development
- An option for standardizing the development platform is to use [containers](https://containers.dev/). 3 dev container configurations are included in the .devcontainer folder for Python 3.8, 3.9 and 3.10.
- Recommended configuration on Windows with WSL:
  - [Visual Studio Code](https://code.visualstudio.com/)
  - [Docker](https://www.docker.com/products/docker-desktop/)
  - [Remote Development - VS Code Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- Opening a development container:
  - Open the project in VS Code
  - Using the Command Palette (F1 or Ctrl+Shift+P)
    - \>Dev Containers: Reopen in Container
    - Select the Python version
    - ???
    - Profit

![](.devcontainer/howto.gif)

## POCs
**POC**: Scott Swan @scott.swan

**DPOC**: Jason Summers @jason.e.summers
