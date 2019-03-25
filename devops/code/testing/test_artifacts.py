import json, pytest, pandas as pd
from azureml.core import Workspace
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core import Experiment
from azureml.core import Run

ws = None

@pytest.fixture()
def get_ws_config():
    global config, ws
    ws = Workspace.from_config()

def test_registered_model(get_ws_config):
    model_list = Model.list(ws)
    model_present = False
    for model in model_list:
        if model.name == "model.pkl":
            model_present = True
            break
    assert model_present, "Model Registered"

def test_registered_model_tags(get_ws_config):
    model_list = Model.list(ws, tags = {"area": "predictive maintenance"})
    assert len(model_list) > 0, "Description about Predictive Maintenance"

def test_scoring_image_present(get_ws_config):
    image_list = Image.list(ws, model_name="model.pkl")
    assert len(image_list) > 0, "Image deployed with model.pkl"

def test_registered_model_metric(get_ws_config):
    try:
        with open("aml_config/run_id.json") as f:
            config = json.load(f)
            new_model_run_id = config["run_id"]
            if new_model_run_id != "":
                experiment_name = config["experiment_name"]
                exp = Experiment(workspace = ws, name = experiment_name)
                model_list = Model.list(ws, tags = {"area": "predictive maintenance"})
                production_model = model_list[0]
                run_list = exp.get_runs()
                new_model_run = Run(exp, run_id=new_model_run_id) 
                new_model_metric = new_model_run.get_metrics().get('accuracy')
                assert new_model_metric > 0.85, "Above 85% accuracy"
    except FileNotFoundError:
        print("No new model registered to test")