# ML flow  concept

### MLflow we use to tracking the eexperiment

### Run these comand for come to vs code from cmd
```
mkdir mlflow  #filename
cd mlflow
code .
```
## in base terminal 
```
conda create -p env python=3.8 -y source activate ./env
```
## In cmd terminal
```
conda create -p env python=3.8 -y conda activate <absloute path of env>
```
#### shortcuts
```
conda create -p env python=3.8 -y && source activate ./env
```
## in base terminal for creating file
```
touch requirements_mlflow.txt
```
```
pip install -r requirements_mlflow.txt
```
```
pip list  #use this to know all install file
```

<write the code in python file>
```
python first_train.py
```
```
mlflow ui  #use for knowing the ml tracks in server
```
```
mlflow dvc
```


# for activating the environments
```
conda activate <path_of_env>--(CMD)
```
```
source activate ./env(bash terminal)---(git base)
```