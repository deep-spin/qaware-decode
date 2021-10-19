Quality Aware Decoding
===

A repository for experiments in quality-aware decoding.

# Setup 

Start by installing the correct version of pytorch for your system.
The rest of the requirements can be installed by running
```
pip install -r requirements.txt
```

# Running experiments with ducttape

Experimentation is based on [ducttape](https://github.com/jhclark/ducttape).
Start by installing it. I recommend installing version [version 0.5](https://github.com/CoderPat/ducttape/releases/tag/v0.5)

The experiments are organized into two files 

* `tapes/main.tape`: This contains the task definitions. It's where you should add new tasks and functionally or edit previously defined ones.
* `tapes/EXPERIMENT_NAME.tconf`: This is where you define the variables for experimentation, as well as which tasks to run.

To start off, we recommend creating you own copy of `tapes/iwsl14.tconf`. 
This file is organized into two parts: (1) the variable definitions at the `global` block (2) the plan definition

To start off, you need to edit the variables to correspond to paths in your file systems. 
Examples include the `$repo` variable and the data variables.

Then try running one of the existing plans by executing

```bash
ducttape tapes/main.tape -C $my_tape -p Baseline -j $num_jobs
```

`$num_jobs` corresponds to the number of jobs to run in parallel. `num_jobs=8` will run 8 branches in parallel if possible (correspondely using 8 GPUs)