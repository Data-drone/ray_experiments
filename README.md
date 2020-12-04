# Ray Experiments

Experiments with BAIR Ray Project

# Starting an image

adding extra 8265 port for ray dashboard

```{bash}

docker run --gpus all --cpus="15" --name ray_test -d -it -p 8900:8888 -p 8265:8265 - -m 30g -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes -v /home/brian/Workspace:/home/jovyan/work --ipc=host -v /media/brian/extra_14:/home/jovyan/work/external_data datadrone/deeplearn_pytorch:latest

```

# Classification Dataset

This is based on the classification dataset from the Kaggle competition: 
https://www.kaggle.com/c/santander-customer-transaction-prediction/


### To Dos

Lets shuffle out the explainability bits.
Focus more on the AutoML bits once we work out a subset of features to do the modelling bit.
Can use the kaggle forum to just put in the best features rather than discovering myself?
Need to look 
### Extra Notes

Needed to uninstall dataclasses library due to issues with Python 3 and dataclasses.

