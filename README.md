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

## Contribution
- the project structure is as follows...
```
jgleeson@daml:$ tree .
.
├── README.md
├── docs
├── pyproject.toml
├── src
│   └── daml
│       ├── __init__.py
│       └── helloworld.py
├── tests
│   └── test_helloworld.py
└── tox.ini

4 directories, 6 files
jgleeson@daml:$
```
- reference the JATIC guidelines for stucture: https://jatic.pages.jatic.net/docs/sdp/Software%20Requirements/#project-structure

- however, generally a `some_module.py` in the `src` folder will need a corresponding `test_some_module.py` in the `tests` folder and execution of tox results in ideally 100% code coverage output. Sub 100% code coverage should be explicitly called out and justified in a merge request.

## Usage
TODO

## POCs
**POC**: Scott Swan @scott.swan

**DPOC**: Jason Summers @jason.e.summers
