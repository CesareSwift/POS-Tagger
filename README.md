# Instructions

Follow the assignment specification. You may use the same Python environment that you created for Lab of Week 8,
or create a new environment as discussed below (instructions are identical to Lab of Week 8).

## Install Requirements
Before installing packages, it is recommended to configure a virtual environment using
[conda](https://docs.conda.io/en/latest/miniconda.html) or [venv](https://docs.python.org/3/library/venv.html).

### Using conda
Create a conda environment and activate it
```
conda create -n [ENV_NAME] python=3.8
conda activate [ENV_NAME]
```

### Using venv
This command will create the environment in the current folder (you need to have Python 3 installed):
```
python -m venv [ENV_NAME]
```
Activate the environment:
```
source [ENV_NAME]/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```
### Details
We explore problems of a Part-of-Speech (POS) tagger from word vectors, which is a tool of tagging a word in a text with its part of speech . We first follow the default task to build a baseline classifier and a POS tagger based on static and contextualised word embeddings. Then we use GPT2 to implement new contextualised word embedding with the default tagger. Next, based on the evaluation of our tagger, we mainly discuss the effect of combination functions and the mistakes which the tagger makes. 
