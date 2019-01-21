# Deploy model as web service


import os 
os.chdir('VS_code/deep_learning/')

# Initialize a Workspace object from the existing workspace you created in 
# the Prerequisites step. Workspace.from_config() creates a workspace object from the details stored in config.json.
from azureml.core.workspace import Workspace

# make sure that you followed the instructions in the readme file to create the config.json file for this step
ws = Workspace.from_config()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')
      
# Once you have your trained model, you can deploy the model on Azure. In this tutorial, we will deploy the model as a web service in Azure Container Instances (ACI). For more information on deploying models using Azure ML, refer here.
# Create scoring script

# First, we will create a scoring script that will be invoked by the web service call. Note that the scoring script must have two required functions:

#     init(): In this function, you typically load the model into a global object. This function is executed only once when the Docker container is started.
#     run(input_data): In this function, the model is used to predict a value based on the input data. The input and output typically use JSON as serialization and deserialization format, but you are not limited to that.

# Refer to the scoring script pytorch_score.py for this tutorial. Our web service will use this file to predict whether an image is an ant or a bee. When writing your own scoring script, don't forget to test it locally first before you go and deploy the web service.


# Create environment file

# Then, we will need to create an environment file (myenv.yml) that specifies all of the scoring script's package dependencies. This file is used to ensure that all of those dependencies are installed in the Docker image by Azure ML. In this case, we need to specify azureml-core, torch and torchvision.

from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(pip_packages=['azureml-defaults', 'torch', 'torchvision'])

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
    
print(myenv.serialize_to_string())


# Configure the container image

# Now configure the Docker image that you will use to build your ACI container.

from azureml.core.image import ContainerImage


image_config = ContainerImage.image_configuration(execution_script='pytorch_score.py', 
                                                  runtime='python', 
                                                  conda_file='myenv.yml',
                                                  description='Image with hymenoptera model')

# Configure the ACI container

# We are almost ready to deploy. Create a deployment configuration file to specify the number of CPUs and gigabytes of RAM needed for your ACI container. While it depends on your model, the default of 1 core and 1 gigabyte of RAM is usually sufficient for many models.

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={'data': 'hymenoptera',  'method':'transfer learning', 'framework':'pytorch'},
                                               description='Classify ants/bees using transfer learning with PyTorch')

# Deploy the registered model

# Get the registered model from your AML workspace
from azureml.core.model import Model
model = Model(workspace = ws, name = 'pytorch-hymenoptera')

# Finally, let's deploy a web service from our registered model. Deploy the web service using the ACI config and image config files created in the previous steps. We pass the model object in a list to the models parameter. If you would like to deploy more than one registered model, append the additional models to this list.

from azureml.core.webservice import Webservice

service_name = 'aci-hymenoptera'
service = Webservice.deploy_from_model(workspace=ws,
                                       name=service_name,
                                       models=[model],
                                       image_config=image_config,
                                       deployment_config=aciconfig,)

service.wait_for_deployment(show_output=True)
print(service.state)

# If your deployment fails for any reason and you need to redeploy, make sure to delete the service before you do so: service.delete()

# Tip: If something goes wrong with the deployment, the first thing to look at is the logs from the service by running the following command:

service.get_logs()

# Get the web service's HTTP endpoint, which accepts REST client calls. This endpoint can be shared with anyone who wants to test the web service or integrate it into an application.

print(service.scoring_uri)

# Test the web service

# Finally, let's test our deployed web service. We will send the data as a JSON string to the web service hosted in ACI and use the SDK's run API to invoke the service. Here we will take an image from our validation data to predict on.

import os, json
from PIL import Image
import matplotlib.pyplot as plt

plt.imshow(Image.open('test_img.jpg'))

import torch
from torchvision import transforms
    
def preprocess(image_file):
    """Preprocess the input image."""
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image.numpy()

input_data = preprocess('test_img.jpg')
result = service.run(input_data=json.dumps({'data': input_data.tolist()}))
print(result)

# Clean up

# Once you no longer need the web service, you can delete it with a simple API call.

service.delete()

