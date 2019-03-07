import os, json,sys
from azureml.core import Workspace
from azureml.core import Run
from azureml.core import Experiment
from azureml.core.model import Model

from azureml.core.runconfig import RunConfiguration

# Get workspace
ws = Workspace.from_config()

# Get the latest evaluation result 
try:
    with open("aml_config/run_id.json") as f:
        config = json.load(f)
    if not config["run_id"]:
        raise Exception('No new model to register as production model perform better')
except:
    print('No new model to register as production model perform better')
    #raise Exception('No new model to register as production model perform better')
    sys.exit(0)

run_id = config["run_id"]
experiment_name = config["experiment_name"]
exp = Experiment(workspace = ws, name = experiment_name)

run = Run(experiment = exp, run_id = run_id)
names=run.get_file_names
names()
print('Run ID for last run: {}'.format(run_id))
model_local_dir="model"
os.makedirs(model_local_dir,exist_ok=True)

# Download Model to Project root directory
model_name= 'model.pkl'
run.download_file(name = './outputs/'+model_name, 
                       output_file_path = './model/'+model_name)
print('Downloaded model {} to Project root directory'.format(model_name))
os.chdir('./model')
model = Model.register(model_path = model_name, # this points to a local file
                       model_name = model_name, # this is the name the model is registered as
                       tags = {'area': "predictive maintenance", 'type': "automl", 'run_id' : run_id},
                       description="Model for predictive maintenance dataset",
                       workspace = ws)
os.chdir('..')
print('Model registered: {} \nModel Description: {} \nModel Version: {}'.format(model.name, model.description, model.version))

# Remove the evaluate.json as we no longer need it
# os.remove("aml_config/evaluate.json")

# Writing the registered model details to /aml_config/model.json
model_json = {}
model_json['model_name'] = model.name
model_json['model_version'] = model.version
model_json['run_id'] = run_id
with open('aml_config/model.json', 'w') as outfile:
  json.dump(model_json,outfile)
 
