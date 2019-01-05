# Introduction 

This repository offers content for a 3 day single scenario deep-dive training outline that has been used for the (ML) track of the Partner AI airlifts in the spring of 2019

This material should enable you to learn how to combine the strengths of Azure Databricks (ADB) for data wrangling and ML model building and training on big data, with the strengths of Azure Machine Learning (AML) services for ML Experimentation, Management, Deployment, and DevOps. 

For this purpose, we will use a real-world scenario for anomaly detection and predictive maintenance on factory floors. This use-case scenario was selected based on popular demand by the field, and its applicability across industry verticals.  

The scenario involves structured as well as unstructured time series data.  The structured time series data will consist of telemetry data collected by sensors on many manufacturing machines. The unstructured time series data will be video recordings of cameras surveilling factory floors. The telemetry data will mainly be used on the first 2 days of the airlift, to show the integration of ADB and AML for classical ML. The video data will be used on the third day, to extend our use-case to detecting anomalies (e.g. accidents) in surveillance videos of factory floors, invoking deep learning on GPUs with AML compute.

# Getting Started

## 1. Create Azure Databricks Cluster:

> If you participating in this course in the classroom, you can skip this step, because we will have already created an Azure Databricks cluster for you. We are leaving this step of the instructions here, in case this is your entrypoint to training content.

Select New Cluster and fill in following detail:
 - Cluster name: _yourclustername_
 - Databricks Runtime: Any **non ML** runtime (non ML 4.x, 5.x)
 - Python version: **3**
 - Workers: 2 or higher.  

These settings are only for using Automated Machine Learning on Databricks.
 - Max. number of **concurrent iterations** in Automated ML settings is **<=** to the number of **worker nodes** in your Databricks cluster.
 - Worker node VM types: **Memory optimized VM** preferred. 
 - Uncheck _Enable Autoscaling_


It will take few minutes to create the cluster. Please ensure that the cluster state is running before proceeding further.

## 2.	Install Azure ML with Automated ML SDK on your Azure Databricks cluster

- Select Import library

- Source: Upload Python Egg or PyPI

- PyPi Name: **azureml-sdk[automl_databricks]**


## 3.	Latest releases
## 4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://www.visualstudio.com/en-us/docs/git/create-a-readme). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)