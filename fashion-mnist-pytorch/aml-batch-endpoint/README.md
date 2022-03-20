# Creating batch endpoints in Azure ML

For a detailed explanation of the code, check out the accompanying blog post: [Creating batch endpoints in Azure ML](https://bea.stollnitz.com/blog/batch-endpoint/).

You can find below the steps needed to create and invoke the endpoints in this project.


## Azure ML setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-36386-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-36386-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-36386-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace." 
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-36386-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-36386-bstollnitz).
* In a terminal window, login to Azure by executing `az login --use-device-code`. 
* Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
* Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-36386-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Although not essential to run the code in this post, I highly recommend installing the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai).


## Create the model

Execute the following command:

```
az ml model create -f cloud/model.yml
```


## Create the cluster

Execute the following command:

```
az ml compute create -f cloud/cluster-cpu.yml
```


## Create the endpoint

Execute the following commands, replacing `<ENDPOINT>` with the name you choose for your endpoint.

```
az ml batch-endpoint create -f cloud/endpoint/endpoint.yml --name <ENDPOINT>
az ml batch-deployment create -f cloud/endpoint/deployment.yml --set-default --endpoint-name <ENDPOINT>
```


## Invoke the endpoint

Execute the following command, replacing `<ENDPOINT>` with the name you choose for your endpoint.

```
az ml batch-endpoint invoke --input-local-path sample-request --name <ENDPOINT>
```


## Get the prediction results

Go to the [Azure ML portal](https://ml.azure.com), click on "Endpoints," "Batch endpoints," and click on the name of the endpoint. Then click on "Runs," and on the latest run, which is displayed at the top. Once the run has completed, write click on the circle that says "score," and choose "Access data." This will take you to the blob storage location where the prediction results are located.
