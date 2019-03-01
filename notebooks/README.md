# LearnAI - CustomAI Airlift

## Part 1: Azure Databricks

> **Important:** If you participating in this course in the classroom, you can skip this step, because we will have already created an Azure Databricks cluster for you. We are leaving this step of the instructions here, in case this is your entrypoint to training content.

### 1. Create Azure Databricks Cluster

Select New Cluster and fill in following detail:

- Cluster name: _yourclustername_
- Databricks Runtime: Any **non-ML** runtime (4.x, 5.x)
- Python version: **3**
- Workers: 2 or higher.  

These settings are only for using Automated Machine Learning on Databricks.

- Max. number of **concurrent iterations** in Automated ML settings is **<=** to the number of **worker nodes** in your Databricks cluster.
- Worker node VM types: **Memory optimized VM** preferred.
- Uncheck _Enable Autoscaling_

It will take few minutes to create the cluster. Please ensure that the cluster state is running before proceeding further.

### 2. Install Azure ML with Automated ML SDK on your Azure Databricks cluster

- Select Import library
- Source: Upload Python Egg or PyPI
- PyPi Name: **azureml-sdk[automl_databricks]**