# mhazili-ai

Mhazili is an advanced AI-powered desktop agent designed to automate and streamline your workflow through intelligent task management and automation. This system leverages cutting-edge AI technologies to understand and execute complex tasks, making your desktop experience more efficient and productive.

## Features

- **Intelligent Task Automation**: Automatically handles repetitive tasks and workflows
- **Smart Screenshot Analysis**: Processes and analyzes screenshots using AI vision capabilities
- **Contextual Understanding**: Maintains context awareness for better task execution
- **Customizable Workflows**: Adaptable to different user needs and preferences
- **Secure Environment**: Local processing with secure handling of sensitive data

## System Requirements

- Python 3.8 or higher
- Windows 10/11 operating system
- Minimum 8GB RAM
- Stable internet connection for AI model access

## Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**

    *   On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    *   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**

    Create a file named `.env` in the root directory of the project. This file will contain your environment variables. You will need to add the necessary variables for the system to run. (e.g., API keys, database connection strings, etc.)

## Running the system

1.  **Activate the virtual environment** (if not already activated):

    *   On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    *   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

2.  **Run the main script:**

    ```bash
    python main.py
    ```
