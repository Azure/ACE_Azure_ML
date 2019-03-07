import numpy
import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice

# Get workspace
ws = Workspace.from_config()

# Get the ACI Details
try:
    with open("aml_config/aci_webservice.json") as f:
        config = json.load(f)
except:
    print('No new model, thus no deployment on ACI')
    #raise Exception('No new model to register as production model perform better')
    sys.exit(0)

service_name = config['aci_name']
# Get the hosted web service
service=Webservice(name = service_name, workspace =ws)

# Input for Model with all features
input_j = [[1.62168882e+02, 4.82427351e+02, 1.09748253e+02, 4.32529303e+01, 3.52377597e+01, 4.37307613e+01, 1.15729573e+01, 4.27624778e+00, 1.68042813e+02, 4.61654301e+02, 1.03138200e+02, 4.08555785e+01, 1.80809993e+01, 4.85402042e+01, 1.09373285e+01, 4.18269355e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.07200000e+03, 5.64000000e+02, 2.22900000e+03, 9.84000000e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.03000000e+02, 6.63000000e+02, 3.18300000e+03, 3.03000000e+02, 5.34300000e+03, 4.26300000e+03, 6.88200000e+03, 1.02300000e+03, 1.80000000e+01]]
print(input_j)
test_sample = json.dumps({'data': input_j})
test_sample = bytes(test_sample,encoding = 'utf8')
try:
    prediction = service.run(input_data = test_sample)
    print(prediction)
except Exception as e:
    result = str(e)
    print(result)
    raise Exception('ACI service is not working as expected')
 
