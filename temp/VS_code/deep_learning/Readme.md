
# Train, hyperparameter tune, and deploy with PyTorch

In this tutorial, you will train, hyperparameter tune, and deploy a PyTorch model using the Azure Machine Learning (Azure ML) Python SDK.

This tutorial will train an image classification model using transfer learning, based on [PyTorch's Transfer Learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). The model is trained to classify ants and bees by first using a pretrained ResNet18 model that has been trained on the [ImageNet](http://image-net.org/index) dataset.

We recommend completing this lab in the following order:

1. Create or connect to a AML Workspace: `create_workspace.py`
1. Train a Deep Neural network on the task: `basic.py`

After a break:
1. Perform hyper parameter tuning: `hyperparameter_tuning.py`
1. Create a webservice: `deployment.py`
1. The your webservice: `test_webservice.py`
