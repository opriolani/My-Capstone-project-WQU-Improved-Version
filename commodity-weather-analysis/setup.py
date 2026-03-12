from setuptools import setup, find_packages

setup(
    name="commodity-weather-analysis",
    version="2.0.0",
    description="Price relationships and weather impact analysis on closely related commodity pairs",
    author="[Your Name]",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "statsmodels>=0.13.5",
        "pmdarima>=2.0.3",
        "arch>=5.3.1",
        "scikit-learn>=1.1.0",
        "xgboost>=1.7.0",
        "plotly>=5.11.0",
        "yfinance>=0.2.12",
        "pyyaml>=6.0",
    ],
)
