# Creating online managed endpoints in Azure ML

For a detailed explanation of the code, check out the accompanying blog post: [Creating managed online endpoints in Azure ML](https://bea.stollnitz.com/blog/managed-endpoint/).

You can find below the steps needed to create and invoke the endpoints in this project.


## Azure setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-31508-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-31508-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-31508-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-31508-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-31508-bstollnitz).
* In a terminal window, login to Azure by executing `az login --use-device-code`. 
* Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
* Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-31508-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Although not essential to run the code in this post, I highly recommend installing the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai).


## Create the models

Execute the following commands:

```
az ml model create -f cloud/model-1.yml
az ml model create -f cloud/model-2.yml
```


## Create and invoke endpoints

Execute the following commands, replacing `<ENDPOINTX>` with the name you choose for each endpoint.


### Endpoint 1

```
az ml online-endpoint create -f cloud/endpoint-1/endpoint.yml --name <ENDPOINT1>
az ml online-deployment create -f cloud/endpoint-1/deployment.yml --all-traffic --endpoint-name <ENDPOINT1>
az ml online-endpoint invoke --request-file sample-request/sample_request.json -n <ENDPOINT1>
```

### Endpoint 2

```
az ml online-endpoint create -f cloud/endpoint-2/endpoint.yml --name <ENDPOINT2>
az ml online-deployment create -f cloud/endpoint-2/deployment.yml --all-traffic --endpoint-name <ENDPOINT2>
az ml online-endpoint invoke --request-file sample-request/sample_request.json -n <ENDPOINT2>
```

### Endpoint 3

```
az ml online-endpoint create -f cloud/endpoint-3/endpoint.yml --name <ENDPOINT3>
az ml online-deployment create -f cloud/endpoint-3/deployment.yml --all-traffic --endpoint-name <ENDPOINT3>
az ml online-endpoint invoke --request-file sample-request/sample_request.json -n <ENDPOINT3>
```

### Endpoint 4

```
az ml online-endpoint create -f cloud/endpoint-4/endpoint.yml --name <ENDPOINT4>
az ml online-deployment create -f cloud/endpoint-4/deployment-blue.yml --all-traffic --endpoint-name <ENDPOINT4>
az ml online-deployment create -f cloud/endpoint-4/deployment-green.yml --endpoint-name <ENDPOINT4>
az ml online-endpoint update --name <ENDPOINT4> --traffic "blue=90 green=10"
az ml online-endpoint invoke --request-file sample-request/sample_request.json -n <ENDPOINT4>
```