# Contributing to CricOptima

Thank you for your interest in contributing! We follow a strict Issue-first workflow.

## 🛠️ Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone** your fork locally.
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Tests** to ensure everything is working:
    ```bash
    pytest
    ```

## 📐 Coding Standards

-   **Style**: We follow PEP 8. Use clear variable names and type hints.
-   **Architecture**: Follow the structure defined in [ARCHITECTURE.md](ARCHITECTURE.md).
-   **Tests**: All new features must include unit tests.

## 🔄 Workflow

> **Important:** Every change requires a GitHub Issue first.

1.  **Create a GitHub Issue** describing your change.
2.  Create a branch from `master` using the naming convention:
    ```
    feat/<description>-<issueID>
    fix/<description>-<issueID>
    docs/<description>-<issueID>
    ```
3.  Commit your changes with conventional messages:
    ```
    feat(scope): description (closes #ID)
    ```
4.  Push to your fork and submit a **Pull Request**.
5.  Ensure your PR includes `Closes #<issueID>` in the description.
6.  **CI must pass** before merge.

## 🐛 Reporting Bugs

Please use the GitHub Issues tab with our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

## 📚 Additional Resources

-   **[CHANGELOG.md](CHANGELOG.md)**: Full release history.
-   **[DEPLOYMENT.md](DEPLOYMENT.md)**: Deployment instructions.

Happy Coding! 🏏
