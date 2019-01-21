
import azureml.core

# Check core SDK version number - based on build number of preview/master.
print("SDK version:", azureml.core.VERSION)

subscription_id = ""
resource_group = "learnai_ai_airlift"
workspace_name = "myADBworkspace"
workspace_region = "westus2"

from azureml.core import Workspace

ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      exist_ok=True)

ws.get_details()

# persist the subscription id, resource group name, and workspace name in aml_config/config.json.
ws.write_config()
