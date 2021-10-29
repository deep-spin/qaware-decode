Quality-Aware Decoding
===

A repository for experiments in quality-aware decoding.

## Setup 

Start by installing the correct version of pytorch for your system.
The rest of the requirements can be installed by running
```bash
pip install -r requirements.txt
```

Experimentation is based on [ducttape](https://github.com/jhclark/ducttape).
Start by installing it. We recommend installing [version 0.5](https://github.com/CoderPat/ducttape/releases/tag/v0.5)

Finally, for experiments involving reranking, [travatar](https://github.com/neubig/travatar). 
Refer to official documentation on how to compile it. 
After installing set the enviroment variable to location of the compiled project
```bash
export TRAVATAR_DIR=/path/of/compiled/travatar/
```

### BLEURT

Evaluating using BLEURT requires tensorflow, making it incompatible with the requirements for the current env.
Therefor the approach took is to use a separate virtual environment for BLEURT. You can do this by

```bash
python -m venv $BLEURT_ENV
source $BLEURT_ENV/bin/activate
pip install git+git://github.com/google-research/bleurt.git@master
pip install --force-reinstall tensorflow-gpu
```
Then set the `bleurt_env` variable in the tconf.
You also need to download one of the BLEURT-20 models and set the `bleurt_dir` variable to it

### OpenKiwi

Similar the BLEURT, OpenKiwi has many dependencies that are incompatible with the rest of the environment.
Therefor we also create an environment specific for OpenKiwi

In order to set it up do:

```bash
python -m venv $OPENKIWI_ENV
source openkiwi_venv/bin/activate
# install pytorch your prefered way 
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
wget https://unbabel-experimental-models.s3.amazonaws.com/openkiwi/openkiwi-2.1.0-py3-none-any.whl
pip install openkiwi-2.1.0-py3-none-any.whl
pip install adapter-transformers==1.1.0
```
Then set the `bleurt_env` variable in the tconf.

It also requires a specific a specific model that was trained with MQM data for the WMT 2021 shared task. 
```bash
wget https://unbabel-experimental-models.s3.amazonaws.com/openkiwi/model_epoch%3D02-val_PEARSON%3D0.79.ckpt -O $OPENKIWI_MODEL
```

This model needs to set in the tconf in the variable `openkiwi_model`.

## Running Experiments


The experiments are organized into two files 

* `tapes/main.tape`: This contains the task definitions. It's where you should add new tasks and functionally or edit previously defined ones.
* `tapes/EXPERIMENT_NAME.tconf`: This is where you define the variables for experiments, as well as which tasks to run.

To start off, we recommend creating you own copy of `tapes/iwsl14.tconf`. 
This file is organized into two parts: (1) the variable definitions at the `global` block (2) the plan definition

To start off, you need to edit the variables to correspond to paths in your file systems. 
Examples include the `$repo` variable and the data variables.

Then try running one of the existing plans by executing

```bash
ducttape tapes/main.tape -C $my_tconf -p Baseline -j $num_jobs
```

`$num_jobs` corresponds to the number of jobs to run in parallel. `num_jobs=8` will run 8 branches in parallel if possible (correspondely using 8 GPUs)