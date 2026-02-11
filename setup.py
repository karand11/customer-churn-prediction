"""
Setup file for customer churn prediction package
"""

from setuptools import setup, find_packages

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Karan Dhanawade",
    author_email="karan.dhanawade10@gmail.com",
    description="Machine learning project for predicting customer churn",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/karand11/customer-churn-prediction.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'jupyter>=1.0.0',
        'joblib>=1.3.2',
        'imbalanced-learn>=0.11.0',
        'xgboost>=1.7.6',
        'plotly>=5.17.0',
    ],
)

