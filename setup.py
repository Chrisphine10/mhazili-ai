from setuptools import setup, find_packages

setup(
    name="ai_automation_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "plotly",
        "pandas",
        "PyYAML",
        "openai",
        "pyautogui",
        "pynput",
        "psutil",
        "pywin32",
    ],
) 