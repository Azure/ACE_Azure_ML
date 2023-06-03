# Introduction

In this course, we will implement a Continuous Integration (CI)/Continuous Delivery (CD) pipeline for Anomaly Detection and Predictive Maintenance applications. For developing an AI application, there are frequently two streams of work:

1. Data Scientists building machine learning models
2. App developers building the application and exposing it to end users to consume

In short, the pipeline is designed to kick off for each new commit, run the test suite, if the test passes takes the latest build, packages it in a Docker container and then deploys to create a scoring service as shown below.

![Architecture](../images/architecture.png)

## Modules Covered

The goal of this course is to cover the following modules:

* Introduction to CI/CD
* Create a CI/CD pipeline using Azure
* Customize a CI/CD pipeline using Azure
* Learn how to develop a Machine Learning pipeline to update models and create service
