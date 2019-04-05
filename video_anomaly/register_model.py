import os, sys
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import get_run
import json

# initialize workspace from config.json
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

try:
    run = get_run(experiment=exp, run_id=run_id, rehydrate=True)
except:
    print("Dir you replace the run_id in the script with that of your hyperdrive run?")
    raise

# children = run.get_children()

# best_metric = 1.0
# best_run_id = ""

# for child in children:
#     status = child.get_status()
    
#     if status == 'Completed':
#         if child.get_metrics()['val_loss'][-1] < best_metric:
#             best_metric = child.get_metrics()['val_loss'][-1]
#             best_run_id = child.get_details()['runId']

# # get the best run            
# best_run = get_run(experiment=exp, run_id=best_run_id, rehydrate=True)

# register the model
model = run.register_model(model_name='prednet', 
                                model_path='outputs')

# Writing the registered model details to /aml_config/model.json
model_json = {}
model_json['model_name'] = model.name
model_json['model_version'] = model.version
model_json['run_id'] = run_id
with open('aml_config/model.json', 'w') as outfile:
  json.dump(model_json, outfile)
 