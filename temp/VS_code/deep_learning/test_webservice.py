# Initialize a Workspace object from the existing workspace you created in 
# the Prerequisites step. Workspace.from_config() creates a workspace object from the details stored in config.json.
from azureml.core.workspace import Workspace

import os 
os.chdir('VS_code/deep_learning/')

# make sure that you followed the instructions in the readme file to create the config.json file for this step
ws = Workspace.from_config()

from azureml.core.webservice import AciWebservice

service_name = 'aci-hymenoptera'
service = AciWebservice(workspace=ws, name=service_name)

print(service.serialize)

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
plt.show(block=False)

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
    image = image.clone().detach()
    image = image.unsqueeze(0)
    return image.numpy()

input_data = preprocess('test_img.jpg')
result = service.run(input_data=json.dumps({'data': input_data.tolist()}))
print(result)

# Clean up

# Once you no longer need the web service, you can delete it with a simple API call.

service.delete()

