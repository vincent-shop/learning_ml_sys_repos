# learning-ml-sys-repos

[![Release](https://img.shields.io/github/v/release/vincentzhong/learning_ml_sys_repos)](https://img.shields.io/github/v/release/vincentzhong/learning_ml_sys_repos)
[![Build status](https://img.shields.io/github/actions/workflow/status/vincentzhong/learning_ml_sys_repos/main.yml?branch=main)](https://github.com/vincentzhong/learning_ml_sys_repos/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vincentzhong/learning_ml_sys_repos/branch/main/graph/badge.svg)](https://codecov.io/gh/vincentzhong/learning_ml_sys_repos)
[![Commit activity](https://img.shields.io/github/commit-activity/m/vincentzhong/learning_ml_sys_repos)](https://img.shields.io/github/commit-activity/m/vincentzhong/learning_ml_sys_repos)
[![License](https://img.shields.io/github/license/vincentzhong/learning_ml_sys_repos)](https://img.shields.io/github/license/vincentzhong/learning_ml_sys_repos)

A repository for learning ML systems and exploring ML system repositories.

- **Github repository**: <https://github.com/vincentzhong/learning_ml_sys_repos/>
- **Documentation** <https://fpgmaas.github.io/example-project/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:vincentzhong/learning_ml_sys_repos.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

### 5. Clone with Submodules

This repository uses git submodules to include external project repositories for code examples. When cloning the repository, include submodules:

```bash
git clone --recurse-submodules git@github.com:vincentzhong/learning_ml_sys_repos.git
```

If you've already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### Adding a New Project Submodule

For each project we add code examples for, we add it as a git submodule in the `examples/` directory:

1. Create a subdirectory for the project in `examples/` (e.g., `examples/miles/`)
2. Add the project repository as a submodule:
   ```bash
   git submodule add <repository-url> examples/<project-name>/<project-name>
   ```
   Example:
   ```bash
   git submodule add https://github.com/radixark/miles.git examples/miles/miles
   ```
3. Commit the submodule addition:
   ```bash
   git add .gitmodules examples/<project-name>/
   git commit -m "Add <project-name> as submodule"
   ```

This pattern keeps external project code separate from our example code while allowing easy reference and updates.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/vincentzhong/learning_ml_sys_repos/settings/secrets/actions/new).
- Create a [new release](https://github.com/vincentzhong/learning_ml_sys_repos/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
