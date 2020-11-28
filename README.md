# Ray Experiments

Experiments with BAIR Ray Project

# Starting an image

adding extra 2100 port for ray dashboard

```{bash}

docker run --gpus all --name ray_test -d -it -p 8900:8888 -p 2100:2100 -e JUPYTER_ENABLE_LAB=yes -v /home/brian/Workspace:/home/jovyan/work --ipc=host -v /media/brian/extra_14:/home/jovyan/work/external_data datadrone/deeplearn_pytorch:latest

```
