# LearnAI - CustomAI Airlift

## Introduction

This repository offers content for a 3 day single scenario deep-dive training outline that has been used for the (ML) track of the Partner AI airlifts in the spring of 2019

This material should enable you to learn how to combine the strengths of Azure Databricks (ADB) for data wrangling and ML model building and training on big data, with the strengths of Azure Machine Learning (AML) services for ML Experimentation, Management, Deployment, and DevOps.

For this purpose, we will use a real-world scenario for anomaly detection and predictive maintenance on factory floors. This use-case scenario was selected based on popular demand by the field, and its applicability across industry verticals.  

The scenario involves structured as well as unstructured time series data.  The structured time series data will consist of telemetry data collected by sensors on many manufacturing machines. The unstructured time series data will be video recordings of cameras surveilling factory floors. The telemetry data will mainly be used on the first 2 days of the airlift, to show the integration of ADB and AML for classical ML. The video data will be used on the third day, to extend our use-case to detecting anomalies (e.g. accidents) in surveillance videos of factory floors, invoking deep learning on GPUs with AML compute.

## Getting Started

The first part of this be performed with Azure Databricks, the second part of the training will be done with VS Code.

Use the following links to get to the right entry point for the two parts:

- Part 1: Azure Databricks: [Readme](./ADB/Readme.md)
- Part 2: VS Code: [Readme](./VS_code/Readme.md)

## API references

- [Spark Python API docs](http://spark.apache.org/docs/latest/api/python/)
- [AML SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

## Contribute

We invite everybody to contribute.

- Email us with questions an comments: ```learnai@microsoft.com```
- Submit issues and comments in our github repository: ```https://github.com/Azure/LearnAI-CustomAI-Airlift/issues```
