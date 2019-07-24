# Introduction

Welcome to the ACE-team training on Azure Machine Learning (AML) service.

The material presented here is a deep-dive which combine real-world data science scenarios with many different technologies including Azure Databricks (ADB), Azure Machine Learning (AML) Services and Azure DevOps, with the goal of creating, deploying, and maintaining end-to-end data science and AI solutions.

# Anomaly Detection in structured data 

- The data scientist has been tasked to develop a predictive maintenance (PdM) solution for a large set of production machines on a manufacturing floor.  

- The data scientist was asked to create a PdM solution that is executed weekly, to develop a maintenance schedule for the next week. 

- Previous experience suggests that anomalies in the telemetry data collected on each machine are indicative of impending catastrophic failures. The data scientist was therefore asked to include anomaly detection in their solution. 

- The organization also asked for a real-time anomaly detection service, to enable immediate machine inspection before the beginning of the next maintenance cycle.  

> **Note:** Anomaly detection can also be performed on unstructured data.  One example is to detect unusual behavior in videos, like a car driving on a sidewalk, or violation of safety protocols on a manufacturing floor.  If you are interested in this use case, please go to this repo: [https://github.com/Microsoft/MLOps_VideoAnomalyDetection](https://github.com/Microsoft/MLOps_VideoAnomalyDetection)

# Creating a lab environment

Please go to [this page](set_up_lab_environment.md) for instructions on how to set up your lab environment. You must follow these instructions whether you are being provided with a lab account or intend to use your own.

# Agendas

Please go to [this page](/agendas/README.md) to find alternative agendas around the above use-cases. 

# References

- [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/)
- [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service/)
- [AML SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)
- [Azure DevOps](https://azure.microsoft.com/en-us/services/devops/)
- [Example notebooks covering various topics](https://github.com/Azure/MachineLearningNotebooks)

# Pre-requisites

### Knowledge/Skills

You will need this basic knowledge:
1. Basic data science and machine learning concepts.
1. Moderate skills in coding with Python and machine learning using Python. 
1. Familiarity with Jupyter Notebooks and/or Databricks Notebooks. 
1. Familiarity with Azure databricks.
1. Basic skills using Git version control.

If you do not have any of the above pre-requisites, please find below links:
1.	To Watch: [Data Science for Beginners](https://www.youtube.com/watch?v=gNV9EqwXCpw)
1.	To Watch: [Get Started with Azure Machine Learning](https://www.youtube.com/watch?v=GBDSBInvz08)
1.	To Watch: [Python for Data Science: Introduction](https://www.youtube.com/watch?v=-Rf4fZDQ0yw&list=PLjgj6kdf_snaw8QnlhK5f3DzFDFKDU5f4)
1.	To Watch: [Jupyter Notebook Tutorial: Introduction, Setup, and Walkthrough](https://www.youtube.com/watch?v=HW29067qVWk&t=564s)
1.	To Do: Go to [https://notebooks.azure.com/] and [create and run a Jupyter notebook](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment) with Python 
1.	To Watch: [Azure Databricks: A brief introduction](https://www.youtube.com/watch?v=cxyUy1bZ9mk&t=1351s)
1. To Read (10 mins): [Git Handbook](https://guides.github.com/introduction/git-handbook/)

### Infrastructure

1. An Azure Subscription (unless provided to you).
1. If you are not provided with a managed lab environment (course invitation will specify), then follow these [instructions for configuring your development environment](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#azure-databricks) prior to the course or if you do it on your own. You will need an Azure Subscription (unless one is provided to you).

# Contribute

We invite everybody to contribute.

- Email us with questions an comments: `learnai@microsoft.com`
- Submit issues and comments in our github repository: [https://github.com/Azure/LearnAI-CustomAI-Airlift/issues](https://github.com/Azure/LearnAI-CustomAI-Airlift/issues)
