# COMP4107: Neural Networks | Group 51

- Andrew Wallace - 101210291 - andrewwallace3@cmail.carleton.ca

## 🧑‍💻 Git Development Workflow

### Prerequisites

- Git is installed locally

### Setup

> Please use the following steps to setup the git repository locally

1. Open your terminal
2. Navigate to the directory you would like the project to live
3. Run:

   `git clone <url>`

   Where `<url>` is the URL of the Git repository

### Starting a New Task

Please use the following steps when starting a new task:

1. Checkout the `main` branch (or whichever branch your team is using as the “base” branch) via:

   `git checkout main`

2. Run the following to get the latest changes from the remote repository:

   `git pull`

3. Run the following command to create a new branch, where `<branch_name>` is the name of the branch you would like to create

   `git checkout -b <branch_name>`

4. Complete your task(s)

## Completing a Task

> Please use the following steps when completing a task

1. Run the following command to add **all** changes to “staging”

   `git add .`

2. Run the following command to commit your changes, where `<short_description>` is a short description of the changes you made

   `git commit -m "<short_description>"`

3. Run the following command to push you changes to the remote repository, where `<branch_name>` is the same name of the branch you created

   `git push -u origin <branch_name>`

4. Navigate to the remote repository (GitHub)
5. Open a new pull request via the “New pull request” button in the “Pull Requests” tab
6. Set the “Compare” branch to your `<branch_name>`
7. Set the “Base” branch to the desired base branch (usually `development` or `main`)
8. Fill out the following:
