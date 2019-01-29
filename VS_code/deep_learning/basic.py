
import os 
# os.chdir('VS_code/deep_learning/')

import azureml.core

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# Diagnostics

# Opt-in diagnostics for better experience, quality, and security of future 
# releases.
from azureml.telemetry import set_diagnostics_collection
set_diagnostics_collection(send_diagnostics=True)

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

# The above code creates a GPU cluster. If you instead want to create a CPU 
# cluster, provide a different VM size to the vm_size parameter, such as 
# STANDARD_D2_V2.


# Train model on the remote compute

# Now that you have your data and training script prepared, you are ready to 
# train on your remote compute cluster. You can take advantage of Azure 
# compute to leverage GPUs to cut down your training time.

# Create a project directory

# Create a directory that will contain all the necessary code from your 
# local machine that you will need access to on the remote resource. This 
# includes the training script and any additional files your training script 
# depends on.

import os

project_folder = './pytorch-hymenoptera'
os.makedirs(project_folder, exist_ok=True)


# Download training data

# The dataset we will use (located here as a zip file) consists of about 120 training images each for ants and bees, with 75 validation images for each class. Hymenoptera is the order of insects that includes ants and bees. We will download and extract the dataset as part of our training script pytorch_train.py
# Prepare training script

# Now you will need to create your training script. In this tutorial, the training script is already provided for you at pytorch_train.py. In practice, you should be able to take any custom training script as is and run it with Azure ML without having to modify your code.

# However, if you would like to use Azure ML's tracking and metrics capabilities, you will have to add a small amount of Azure ML code inside your training script.

# In pytorch_train.py, we will log some metrics to our Azure ML run. To do so, we will access the Azure ML Run object within the script:

# from azureml.core.run import Run
# run = Run.get_context()

# Further within pytorch_train.py, we log the learning rate and momentum parameters, and the best validation accuracy the model achieves:

# run.log('lr', np.float(learning_rate))
# run.log('momentum', np.float(momentum))

# run.log('best_val_acc', np.float(best_acc))

# These run metrics will become particularly important when we begin hyperparameter tuning our model in the "Tune model hyperparameters" section.

# Once your script is ready, copy the training script pytorch_train.py into your project directory for staging.

import shutil
shutil.copy('pytorch_train.py', project_folder)


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



# The script_params parameter is a dictionary containing the command-line arguments to your training script entry_script. Please note the following:

#     We passed our training data reference ds_data to our script's --data_dir argument. This will 1) mount our datastore on the remote compute and 2) provide the path to the training data hymenoptera_data on our datastore.
#     We specified the output directory as ./outputs. The outputs directory is specially treated by Azure ML in that all the content in this directory gets uploaded to your workspace as part of your run history. The files written to this directory are therefore accessible even once your remote run is over. In this tutorial, we will save our trained model to this output directory.

# To leverage the Azure VM's GPU for training, we set use_gpu=True.
# Submit job

# Run your experiment by submitting your estimator object. Note that this call is asynchronous.

run = experiment.submit(estimator)
print(run)

# to get more details of your run
print(run.get_details())

# block until the script has completed training before running more code.
run.wait_for_completion(show_output=True)

