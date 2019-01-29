# Azure Machine Learning Pipelines

In this session, you will be introduced to [Azure Machine Learning Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines).  A secondary goal is to gain some familiarity with using the open-source IDE [VS Code](https://code.visualstudio.com/) for creating data science solutions using Azure Machine Learning services.

Imagine the following real-world scenario: We are collecting data from 100 manufacturing machines.  Four sensors are installed on each machine, to collect telemetry data (voltage, rotation, pressure and vibration measurements). Measurements are collected every hour, to create a structured time series. 

Our goal is to use these data to train a machine learning model for [predictive maintenance](https://gallery.azure.ai/Collection/Predictive-Maintenance-Implementation-Guide-1). That is, we try to predict if a machine is going to fail soon, so that we can schedule maintenance to prevent this failure.

We try to reach this goal in two steps:
1. Preprocess the data and perform data engineering specific for time series data. Here we also perform anomaly detection, to include detected anomalies as features that the predictive maintenance model can use in its prediction.
1. We use [automated ML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml) to automate the process of taking our training data with a defined target feature, and iterating through combinations of algorithms and feature selections to automatically select the best model for our predictive maintenance scenario.

We will define an Azure Machine Learning pipeline to combine these steps into one pipeline.  In another course on Continuous Integration and Continuous Delivery (CI/CD), we will extend this implementation so that the pipeline is run again, e.g. if a significant of new data has been collected, to ensure that our model's predictions are consistent with more recent data.

## Pre-Requisites

- A current installation of Anaconda ([Miniconda](https://conda.io/en/latest/miniconda.html) is sufficient and faster to install).
- Recommended: A current installation of [VS Code](https://code.visualstudio.com/)
- A conda environment with all the dependencies. Use the environment configuration file `environment.yml` in the parent directory. 
  - In VS Code, select `View` in the menu, and select `Integrated Terminal`.
  - Change directory into `VS_Code`. 
  - Type: `conda env create -f environment.yml` (execution can take up to a few minutes). *Note:* If you decide to change your conda environment definition, you can update an existing environment by typing: `conda env update -f environment.yml`

## Azure Machine Learning Pipelines

[Azure Machine Learning Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines) enable data scientists to create and manage multiple simple and complex workflows concurrently. A typical pipeline would have multiple tasks to prepare data, train, deploy and evaluate models. Individual steps in the pipeline can make use of diverse compute options (for example: CPU for data preparation and GPU for training) and languages.

### Creating a Pipeline

A Step is a unit of execution. Step typically needs a target of execution (compute target), a script to execute, and may require script arguments and inputs, and can produce outputs. The step also could take a number of other parameters. Azure Machine Learning Pipelines provides the following built-in Steps:

- PythonScriptStep: Add a step to run a Python script in a Pipeline.
- AdlaStep: Adds a step to run U-SQL script using Azure Data Lake Analytics.
- DataTransferStep: Transfers data between Azure Blob and Data Lake accounts.
- DatabricksStep: Adds a DataBricks notebook as a step in a Pipeline.
- HyperDriveStep: Creates a Hyper Drive step for Hyper Parameter Tuning in a Pipeline.
- EstimatorStep: Adds a step to run Estimator in a Pipeline.
- MpiStep: Adds a step to run a MPI job in a Pipeline.
- HyperDriveStep: Creates a HyperDrive step in a Pipeline.

For our scenario, we will use two `PythonScriptStep` definitions, and link them with a `PipelineData` to pass preprocessed data to the automated ML step.

To make this happen, we create three files:
1. `pipeline.py` this is in a way the `main` function of our pipeline, it orchestrates the execution of individual steps.
1. `anom_detect.py` performs the data preprocessing and anomaly detection.
1. `automl_step.py` takes the preprocessed data to perform automatic feature and algorithm selection.

**pipeline.py** can roughly be split into the following sections:
1. Importing of python modules needed for Azure Machine Learning experimentation, model management, and defining an AML pipeline.
1. Creating an AML workspace and defining and experiment
1. Define a compute target. We are using [AmlCompute](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.amlcompute(class)?view=azure-ml-py) as a compute target. 
1. Connect to the data file and blob store in our workspace to ingest data and transfer data between steps of our pipeline.
1. Validation of Pipeline
1. Running the Pipeline
1. Downloading the results to our local computer for manual inspection.

**anom_detect.py** can be split into the following sections
1. Importing of modules. Note that we are using an open-source 3rd party library, [pyculiarity](https://pypi.org/project/pyculiarity/) for detecting anomalies
1. Method definitions for [running averages](https://en.wikipedia.org/wiki/Moving_average) and performing anomaly detection.
1. Loading data from blob and parsing data columns to ensure the correct data type.
1. We loop over manufacturing machines to process all data we have collected so far.
1. We [pickle](https://docs.python.org/3/library/pickle.html) the data frame that contains the preprocessed data, so that it can be passed to the next step in the pipeline.

**automl_step.py** can be split into the following section:
1. Importing of modules
1. Definition of auxiliary functions
1. Some more feature engineering beyond anomaly detection
1. Configuration of our AutoML experiment
1. Evaluation of our experiment
1. Uploading of the results to our data store, so that we can access them in the master `pipeline.py` file.

## Pipeline Execution

### Create AML workspace

As you may have guessed, the first step is to create an AML workspace.  Do this by modifying the file `config_sample.json` so that it reflects your settings from the previous labs.  Then rename the file to `config.json`.

### Set breakpoints

It would take a while to go through every single line of the code of our AML pipeline, so let's set some breakpoints to highlight essential steps of our pipeline execution.

We recommend to set all the breakpoints listed here first, before you start debugging. While debugging, go through this list again to get a deeper understanding what happens at each of these breakpoints.

- `pipeline.py`
  - `print("Azure Machine Learning Compute attached")` - Go into your AML workspace in the Azure portal and try to find the compute target that you created here. Explore a bit what the properties are and how to check its status.
  - `def_data_store = ws.get_default_datastore()` - try to find this in the Azure portal.
  - `def_blob_store = Datastore(ws, "workspaceblobstore")` - also try to find this in the portal.
  - `print("Anomaly Detection Step created.")` - explore the `anom_detect` script step definition. What is the output? Look into the documentation for AML pipelines and try to find out what the effect of `allow_reuse=True` is. What is the compute_target definition?
  - `print("AutoML Training Step created.")` - what are the inputs here?
  - `print("Pipeline run completed")` - investigate the lines between here and the line previous training step.  Make sure you understand what is happening in each line.  Go into the Azure portal and try to find the pipeline you created.
  - `print('Uploaded the model {} to experiment {}'.format(model_fname, pipeline_run.experiment.name))` - try to find the newly created objects in the `def_data_store`.
  



