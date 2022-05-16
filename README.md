Quality-Aware Decoding
===
[![Python Lint](https://github.com/deep-spin/qaware-decode/actions/workflows/pylint.yml/badge.svg)](https://github.com/deep-spin/qaware-decode/actions/workflows/pylint.yml)


This is the official repository for the paper [Quality-Aware Decoding for Neural Machine Translation](https://arxiv.org/abs/2205.00978).

<hr />

> **Abstract:** *Despite the progress in machine translation quality estimation and evaluation in the last years, decoding in neural machine translation (NMT) is mostly oblivious to this and centers around finding the most probable translation according to the model (MAP decoding), approximated with beam search.  maximum-a-posteriori} (MAP) translation. In this paper, we bring together these two lines of research and propose \emph{quality-aware decoding} for NMT, by leveraging recent breakthroughs in reference-free and reference-based MT evaluation through various inference methods like $N$-best reranking and minimum Bayes risk decoding. We perform an extensive comparison of various possible {candidate generation} and {ranking} methods across four datasets and two model classes and find that quality-aware decoding consistently outperforms MAP-based decoding according  both to state-of-the-art automatic metrics (COMET and BLEURT) and to human assessments.*
<hr />

# The `qaware-decode` package

We provide a package to make quality-aware decoding more accessible to practitioners/researchers trying to improve their MT models.

Start by installing the package with

```bash
git clone https://github.com/deep-spin/qaware-decode.git && cd qaware-decode
pip install -e .
```

This will install the package, plus the necessary dependencies for the COMET-family metrics.
You can also install other metrics with the optional dependency groups
    
```bash
pip install ".[mbart-qe]"
pip install ".[transquest]"
```

Performing quality-aware decoding is as simple as passing the n-best hypothesis list one of the `qaware-decode` commands.
For example, to apply MBR with COMET on an n-best list extracted with `fairseq`, just do

```bash
fairseq-generate ... --nbest $nbest | grep ^H | cut -c 3- | sort -n | cut -f3- > $hyps
qaware-mbr $hyps --src $src -n $nbest > qaware-decode.txt
```

If you pass references, the library wil also perform evaluation of the decoded sentences.

```bash
qaware-mbr $hyps --src $src -n $nbest --refs $refs > qaware-decode.txt
```

## Minimum Bayes Risk (MBR)

To perform MBR, we provide the `qaware-mbr` command. 
You can specify the metric to perform with the `--metric` option.

```bash
qaware-mbr $hyps --src $src -n $nbest --metric bleurt > mbr-decode.txt
```

## N-best Reranking

To perform N-best reranking, we provide the `qaware-rerank` command. 
You can specify the QE metric to use for reranking `--qe-metrics` option.

```bash
qaware-rerank $hyps --src $src -n $nbest --qe-metrics comet_qe \
    > rerank-decode.txt
```

You can also *train* a reranker to use multiple metrics when reranking, as well as the original probabilities given by the model.
To do this you need to have a *dev* set with associated references. You also need [travatar](https://github.com/neubig/travatar) installed. 

To train a reranked, just specify the `--train-reranker` option. 
You can specify what metric to optimize over with `--rerank-metric`.

```bash
qaware-rerank 
    $dev_hyps \
    --src $dev_src \
    --refs $dev_refs \
    --scores $dev_scores \
    --num-samples $nbest \
    --qe-metrics comet_qe mbart_qe \
    --langpair en-de \
    --train-reranker learned_weights.json \
    --rerank-metric comet \
    > /dev/null 

```

Then you can use the learned weights to rerank another set of hypotheses.

```bash
qaware-rerank 
    $hyps \
    --src $src \
    --refs $refs \
    --scores $scores \
    --num-samples $nbest \
    --qe-metrics comet_qe mbart_qe \
    --langpair en-de \
    --weights learned_weights.json \
    > t-rerank-decode.txt
```


# Reproducing the results of the paper

## Setup 

Start by installing the correct version of pytorch for your system.
The rest of the requirements can be installed by running
```bash
pip install -r requirements.txt
```

Experimentation is based on [ducttape](https://github.com/jhclark/ducttape).
Start by installing it. We recommend installing [version 0.5](https://github.com/CoderPat/ducttape/releases/tag/v0.5)

Finally, for experiments involving reranking, [travatar](https://github.com/neubig/travatar). 
Refer to the official documentation on how to compile it. 
After installing set the environment variable to the location of the compiled project
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

`$num_jobs` corresponds to the number of jobs to r
