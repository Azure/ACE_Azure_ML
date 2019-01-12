
# Tune model hyperparameters

import azureml.core

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# Initialize a Workspace object from the existing workspace you created in 
# the Prerequisites step. Workspace.from_config() creates a workspace object from the details stored in config.json.
from azureml.core.workspace import Workspace

# make sure that you followed the instructions in the readme file to create the config.json file for this step
ws = Workspace.from_config()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# Create or Attach existing AmlCompute

# You will need to create a compute target for training your model. In this 
# tutorial, we use Azure ML managed compute (AmlCompute) for our remote 
# training compute resource.

# Creation of AmlCompute takes approximately 5 minutes. If the AmlCompute 
# with that name is already in your workspace, this code will skip the 
# creation process.

# As with other Azure services, there are limits on certain resources 
# (e.g. AmlCompute) associated with the Azure Machine Learning service. 
# Please read this article on the default limits and how to request more 
# quota.
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "gpucluster"

try:
    compute_target = AmlCompute(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                            max_nodes=4,
                                                            idle_seconds_before_scaledown=1800)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())


# Create a project directory

# Create a directory that will contain all the necessary code from your 
# local machine that you will need access to on the remote resource. This 
# includes the training script and any additional files your training script 
# depends on.

import os

project_folder = './VS_code/deep_learning/pytorch-hymenoptera'
os.makedirs(project_folder, exist_ok=True)

# Create an experiment

# Create an Experiment to track all the runs in your workspace for this transfer learning PyTorch tutorial.

from azureml.core import Experiment

experiment_name = 'pytorch-hymenoptera'
experiment = Experiment(ws, name=experiment_name)

# Create a PyTorch estimator

# The Azure ML SDK's PyTorch estimator enables you to easily submit PyTorch training jobs for both single-node and distributed runs. For more information on the PyTorch estimator, refer here. The following code will define a single-node PyTorch job.

from azureml.train.dnn import PyTorch

script_params = {
    '--num_epochs': 30,
    '--output_dir': './outputs'
}

estimator = PyTorch(source_directory=project_folder, 
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='pytorch_train.py',
                    use_gpu=True)

# Now that we've seen how to do a simple PyTorch training run using the SDK, let's see if we can further improve the accuracy of our model. We can optimize our model's hyperparameters using Azure Machine Learning's hyperparameter tuning capabilities.
# Start a hyperparameter sweep

# First, we will define the hyperparameter space to sweep over. Since our training script uses a learning rate schedule to decay the learning rate every several epochs, let's tune the initial learning rate and the momentum parameters. In this example we will use random sampling to try different configuration sets of hyperparameters to maximize our primary metric, the best validation accuracy (best_val_acc).

# Then, we specify the early termination policy to use to early terminate poorly performing runs. Here we use the BanditPolicy, which will terminate any run that doesn't fall within the slack factor of our primary evaluation metric. In this tutorial, we will apply this policy every epoch (since we report our best_val_acc metric every epoch and evaluation_interval=1). Notice we will delay the first policy evaluation until after the first 10 epochs (delay_evaluation=10). Refer here for more information on the BanditPolicy and other policies available.

from azureml.train.hyperdrive import RandomParameterSampling, HyperDriveRunConfig, BanditPolicy, PrimaryMetricGoal, uniform

param_sampling = RandomParameterSampling( {
        'learning_rate': uniform(0.0005, 0.005),
        'momentum': uniform(0.9, 0.99)
    }
)

early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)

hyperdrive_run_config = HyperDriveRunConfig(estimator=estimator,
                                            hyperparameter_sampling=param_sampling, 
                                            policy=early_termination_policy,
                                            primary_metric_name='best_val_acc',
                                            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                            max_total_runs=8,
                                            max_concurrent_runs=4)

# Finally, lauch the hyperparameter tuning job.

# start the HyperDrive run
hyperdrive_run = experiment.submit(hyperdrive_run_config)

# Monitor HyperDrive runs

# Or block until the HyperDrive sweep has completed:

hyperdrive_run.wait_for_completion(show_output=True)

# Find and register the best model

# Once all the runs complete, we can find the run that produced the model with the highest accuracy.

best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
print(best_run)

print('Best Run is:\n  Validation accuracy: {0:.5f} \n  Learning rate: {1:.5f} \n  Momentum: {2:.5f}'.format(
        best_run_metrics['best_val_acc'][-1],
        best_run_metrics['lr'],
        best_run_metrics['momentum'])
     )

# Finally, register the model from your best-performing run to your workspace. The model_path parameter takes in the relative path on the remote VM to the model file in your outputs directory. In the next section, we will deploy this registered model as a web service.

model = best_run.register_model(model_name = 'pytorch-hymenoptera', model_path = 'outputs/model.pt')
print(model.name, model.id, model.version, sep = '\t')
