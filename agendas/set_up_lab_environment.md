# Setting up the lab envirnoment

## Install AML Python SDK with Automated ML SDK on your Azure Databricks cluster

Please follow these [instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#azure-databricks) to setup an Azure Databricks cluster with the Python AML SDK.

## Import course jupyter notebooks

You can either import the jupyter notebooks one my one, or create a databricks archive that contains all the notebooks, to then import the archive.

Follow these simple steps to create a databricks archive

```
cd <root>/presenter
zip -0 -r notebooks notebooks
mv notebooks.zip notebooks.dbc
```

As you can see, a databricks is simply a zip archive with compression level zero.
