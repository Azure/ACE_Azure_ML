import os, json
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.model import Model
import azureml.core
from azureml.core import Run


# Get workspace
ws = Workspace.from_config()

# Paramaterize the matrics on which the models should be compared

# Add golden data set on which all the model performance can be evaluated

# Get the latest run_id 
with open("aml_config/run_id.json") as f:
    config = json.load(f)

new_model_run_id = config["run_id"]
production_model_run_id = ''
experiment_name = config["experiment_name"]
exp = Experiment(workspace = ws, name = experiment_name)


try:
  # Get most recently registered model, we assume that is the model in production. Download this model and compare it with the recently trained model by running test with same data set.
  model_list = Model.list(ws)
  production_model = next(filter(lambda x: x.created_time == max(model.created_time for model in model_list),  model_list))
  production_model_run_id = production_model.tags.get('run_id')
  run_list = exp.get_runs()
  # production_model_run = next(filter(lambda x: x.id == production_model_run_id, run_list))


  # Get the run history for both production model and newly trained model and compare mse
  production_model_run = Run(exp,run_id=production_model_run_id)
  new_model_run = Run(exp,run_id=new_model_run_id)
 
  production_model_metric = production_model_run.get_metrics().get('accuracy')
  new_model_metric = new_model_run.get_metrics().get('accuracy')
  print('Current Production model accuracy: {}, New trained model accuracy: {}'.format(production_model_metric, new_model_metric))

  promote_new_model=False
  if new_model_metric < production_model_metric:
    promote_new_model=True
    print('New trained model performs better, thus it will be registered')
except:
  promote_new_model=True
  print('This is the first model to be trained, thus nothing to evaluate for now')

run_id = {}
# Use the old production_model_run_id as default
run_id['run_id'] = production_model_run_id
# Writing the run id to /aml_config/run_id.json
if promote_new_model:
  run_id['run_id'] = new_model_run_id

run_id['experiment_name'] = experiment_name
with open('aml_config/run_id.json', 'w') as outfile:
  json.dump(run_id,outfile)
 
