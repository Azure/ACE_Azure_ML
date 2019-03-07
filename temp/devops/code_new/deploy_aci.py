import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice

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
print('From image.json, Image used to deploy webservice on ACI: {}\nImage Version: {}\nImage Location = {}'.format(image.name, image.version, image.image_location))

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                            memory_gb=1, 
                                            tags={'area': "pred-maintenance", 'type': "automl"},
                                            description='A sample description')

aci_service_name='aciwebservice'+ datetime.datetime.now().strftime('%m%d%H')

service = Webservice.deploy_from_image(deployment_config=aciconfig,
                                        image=image,
                                        name=aci_service_name,
                                        workspace=ws)

service.wait_for_deployment()
print('Deployed ACI Webservice: {} \nWebservice Uri: {}'.format(service.name, service.scoring_uri))

#service=Webservice(name ='aciws0622', workspace =ws)
# Writing the ACI details to /aml_config/aci_webservice.json
aci_webservice = {}
aci_webservice['aci_name'] = service.name
aci_webservice['aci_url'] = service.scoring_uri
with open('aml_config/aci_webservice.json', 'w') as outfile:
  json.dump(aci_webservice,outfile)
 
