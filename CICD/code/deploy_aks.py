import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice

# Get workspace
ws = Workspace.from_config()

# Get the Image to deploy details
try:
    with open("aml_config/image.json") as f:
        config = json.load(f)
except:
    print('No new model, thus no deployment on ACI')
    #raise Exception('No new model to register as production model perform better')
    sys.exit(0)

image_name = config['image_name']
image_version = config['image_version']

images = Image.list(workspace=ws)
image, = (m for m in images if m.version==image_version and m.name == image_name)
print('From image.json, Image used to deploy webservice: {}\nImage Version: {}\nImage Location = {}'.format(image.name, image.version, image.image_location))

# Check if AKS already Available
try:
    with open("aml_config/aks_webservice.json") as f:
        config = json.load(f)
    aks_name = config['aks_name']
    aks_service_name = config['aks_service_name']
    compute_list = ws.compute_targets()
    aks_target, =(c for c in compute_list if c.name ==aks_name)
    service=Webservice(name =aks_service_name, workspace =ws)
    print('Updating AKS service {} with image: {}'.format(aks_service_name,image.image_location))
    service.update(image=image)
except:
    aks_name = 'aks'+ datetime.datetime.now().strftime('%m%d%H')
    aks_service_name = 'akswebservice'+ datetime.datetime.now().strftime('%m%d%H')
    prov_config = AksCompute.provisioning_configuration(agent_count = 6, vm_size = 'Standard_F2', location='eastus')
    print('No AKS found in aks_webservice.json. Creating new Aks: {} and AKS Webservice: {}'.format(aks_name,aks_service_name))
    # Create the cluster
    aks_target = ComputeTarget.create(workspace = ws, 
                                    name = aks_name, 
                                    provisioning_configuration = prov_config)

    aks_target.wait_for_completion(show_output = True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)

    # Use the default configuration (can also provide parameters to customize)
    aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)

    service = Webservice.deploy_from_image(workspace = ws, 
                                            name = aks_service_name,
                                            image = image,
                                            deployment_config = aks_config,
                                            deployment_target = aks_target)
                            
    service.wait_for_deployment(show_output = True)
    print(service.state)
    print('Deployed AKS Webservice: {} \nWebservice Uri: {}'.format(service.name, service.scoring_uri))



# Writing the AKS details to /aml_config/aks_webservice.json
aks_webservice = {}
aks_webservice['aks_name'] = aks_name
aks_webservice['aks_service_name'] = service.name
aks_webservice['aks_url'] = service.scoring_uri
aks_webservice['aks_keys'] = service.get_keys()
with open('aml_config/aks_webservice.json', 'w') as outfile:
  json.dump(aks_webservice,outfile)
 
