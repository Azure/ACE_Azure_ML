# Deep Learning

## Pre-Requisites

### Software Installation

**Visual Studio Code**

- Binaries available here [https://code.visualstudio.com/](https://code.visualstudio.com/)
- Extensions:
  - Python Extension
  - Azure Machine Learning
  - Azure Account

If you have not used VS Code before, we recommend you visit the *Getting Started* section of the [VS Code documentation](https://code.visualstudio.com/docs).

### Clone Source Repository

To clone the source code and notebooks (including this document) for these hands-on labs, follow these steps:

- press ```ctrl+shift+p```
- type: ```Git: Clone```
- paste the URL of our repository: ```https://github.com/azure/LearnAI-CustomAI-Airlft```

You will find the sources for our **Deep Learning** labs in the subfolder: ```VS_code```.

### Install AML SDK for Python

1. Install the AML SDK for Python
   1. If you don't have an open terminal *inside* of VS Code: ```Terminal->New Terminal```
   2. type: ```conda env create -f environment.yml```
2. Restart VS Code
3. Select the Python interpreter for this conda environment
   1. press ```ctrl+shift+p```
   2. type: ```Python: Select Interpreter```
   3. Select a Python interpreter **3.5** or greater


### Log into your Azure Account

The [Azure Account extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azure-account) was automatically installed in Visual Studio Code during the install step.

Please use it to login by using the `Azure: Sign In` (press ```ctrl+shift+p``` first)command.

To learn more about how to create or filter your Azure subscriptions, here is a quick reference of the commands available:

| Command | Description |
| --- |---|
| `Azure: Sign In`  | Sign in to your Azure subscription.
| `Azure: Sign Out` | Sign out of your Azure subscription.
| `Azure: Select Subscriptions` | Pick the set of subscriptions you want to work with.
| `Azure: Create an Account`  | If you don't have an Azure Account, you can [sign up](https://azure.microsoft.com/en-us/free/?utm_source=campaign&utm_campaign=vscode-azure-account&mktingSource=vscode-azure-account) for one today and receive $200 in free credits.

## 3. Getting started with Azure Machine Learning

### Create an Azure Machine Learning workspace

1. Open the `Azure activity` bar in Visual Studio Code.
1. Open the Azure Machine Learning view.
1. Right-click your Azure subscription and select `Create Workspace`.
1. Specify a name for your new workspace.
1. Select an existing resource group or create a new one using the wizard in the command palette.
1. Hit enter.

![create workspace](./media/createworkspace.gif)

### Create an Azure Machine Learning experiment

This will enable you to keep track of your experiments using Azure Machine Learning

1. Right-click an Azure Machine Learning workspace and select `Create Experiment` from the context menu.
1. Name your experiment and hit enter.

### Attach your folder to your experiment

Right click on an Azure Machine Learning experiment and select `Attach Folder to Experiement` - This will enable associating each of your experiment runs with your experiment so all of your key metrics will be stored in the experiment history and the models you train will get automatically uploaded to Azure Machine Learning and stored with your experimment metrics and logs.

![attach folder](./media/attachfolder.gif)

## Next steps

- To learn how to create and manage compute resources in Azure Machine Learning from within VS Code, see [Create and manage compute targets in Visual Studio Code](manage-compute-aml-vscode.md)
- To learn how to train models and manage your experiments from Visual Studio Code, see [Training models and managing experiments in Visual Studio Code](train-models-aml-vscode.md)
- To learn how to deploy and manage models from Visual Studio Code, see [Deploying and managing models in Visual Studio Code](deploy-models-aml-vscode.md)
