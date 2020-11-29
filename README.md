# Ray Experiments

Experiments with BAIR Ray Project

# Starting an image

adding extra 8265 port for ray dashboard

```{bash}

docker run --gpus all --name ray_test -d -it -p 8900:8888 -p 8265:8265 -e JUPYTER_ENABLE_LAB=yes -v /home/brian/Workspace:/home/jovyan/work --ipc=host -v /media/brian/extra_14:/home/jovyan/work/external_data datadrone/deeplearn_pytorch:latest

```

# Classification Dataset

This is based on the classification dataset from the Kaggle competition: 
https://www.kaggle.com/c/santander-customer-transaction-prediction/

### Extra Notes

Needed to uninstall dataclasses library due to issues with Python 3 and dataclasses.
