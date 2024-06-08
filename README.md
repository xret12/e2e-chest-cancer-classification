# End-to-End Chest Cancer Classification ML Project
- Deep learning: CNN model
- Ops: MLFlow, DVC
- https://www.youtube.com/watch?v=-NOIWzjJK-4&t=18199s&ab_channel=DSwithBappy

## Workflows (TO-DO)
1. Update config.yaml
2. Update secrets.yaml [optional]
3. Update params.yaml
4. Update the entity
5. Update the config manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml


## DAGSHUB Connection for MLFLOW (windows machine)
- SET for windows / EXPORT for linux

```
MLFLOW_TRACKING_URI=https://dagshub.com/xret12/e2e-chest-cancer-classification.mlflow

MLFLOW_TRACKING_USERNAME=xret12

MLFLOW_TRACKING_PASSWORD=c05d9ad81ab52d7ad65f1197d91df0ab2f92d11f

```

## AWS-CICD-Deployment-with-Github-Actions
### 1. Login to AWS console.
### 2. Create IAM user for deployment
    #with specific access

    1. EC2 access : It is virtual machine

    2. ECR: Elastic Container registry to save your docker image in aws


#### Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#### Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
### 3. Create ECR repo to store/save docker image
 Save the repository URI

    521740697242.dkr.ecr.us-east-1.amazonaws.com/e2e-chest-cancer-classif
### 4. Create EC2 machine (Ubuntu)
### 5. Open EC2 and Install docker in EC2 Machine:
    #optional

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker
### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one
### 7. Setup github secrets:
    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = 521740697242.dkr.ecr.us-east-1.amazonaws.com

    ECR_REPOSITORY_NAME = e2e-chest-cancer-classif
