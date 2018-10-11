# Azure ML Pipeline

This project details the end-to-end Machine Learning pipeline built using the Azure Machine Learning service. It includes the following components: <br>
1. Training 
2. Local Testing 
3. Creating Docker Images
4. Creating Deployment on Azure Container Instance (ACI) Service
5. Testing Deployment

## Overview of the ML Pipeline
![Alt text](./extras/ML-Pipeline.png?raw=true "Title")

## Steps to follow to execute sklearn example

1. **Run generate_wsconfig.py**: This contains the subscription id, resource group and worksplace. Output of this script is a config file created under aml_config/config.json <br>
2. **Run train.py**: This contains the training script that will be executed for training your model. This registers the trained model to the workspace. <br>
3. **Run main_train_sklearn.py**: This contains the configuration for creating an experiment, generating a run under the experiment, creating the VM and submitting the training script for execution to the VM. 
4. **Locally download the model**: Download the trained model locally from the VS Code IDE. The Azure extension will enable you to do that. 
5. **Create Docker Image by running create_docker_image.py**: Once the model is tested, docker image can be created using the registered model, prediction script (score.py) and docker config. 
6. **Create Deployment by running create_deployment.py**: The above created docker image can be deployed on the Azure Container Instance (ACI) service. This enables HTTP client requests. 
7. **Test Deployment by running test_deployment.py**: Test the deployment. 
