# Ray Experiments

Experiments with BAIR Ray Project

# Starting an image

8265 port for ray dashboard
8900 port for jupyter lab
6006 port fot tensorboard

```{bash}

docker run --gpus all --cpus="15" --name ray_test -d -it -p 8900:8888 -p 8265:8265 -p 6006:6006 -m 30g -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes -v /home/brian/Workspace:/home/jovyan/work --ipc=host -v /media/brian/extra_14:/home/jovyan/work/external_data datadrone/deeplearn_pytorch:latest

```

# Classification Dataset

This is based on the classification dataset from the Kaggle competition: 
https://www.kaggle.com/c/santander-customer-transaction-prediction/


### To Dos

- Get different kernels setup for different packages to prevent issues with compatabilities
  - https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

- Build out the right features so that AutoML performs better
  - look at diff between distributions between target = 1 and target = 0
  - presence in test vs train sets

- Then shuffle out the explainability bits.

- Focus more on the AutoML bits once we work out a subset of features to do the modelling bit.
  - mljar - https://medium.com/@MLJARofficial/mljar-supervised-automl-with-explanations-and-markdown-reports-36d5104e117
  - pycaret - https://medium.com/analytics-vidhya/first-medium-blog-on-auto-ml-pycaret-a6deb5748fba
  - tpot + cudf - https://medium.com/rapids-ai/two-years-in-a-snap-rapids-0-16-ae797795a5c4
  - ray options - https://medium.com/rapids-ai/30x-faster-hyperparameter-search-with-raytune-and-rapids-403013fbefc5

### Extra Notes

Needed to uninstall dataclasses library due to issues with Python 3 and dataclasses.

