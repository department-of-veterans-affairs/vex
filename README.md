## VEx
Rule-based NLP systems require an extensive lexicon in order to discover concepts in clinical text. Depending on the task, this lexicon may be relatively simple to develop. However, challenges in developing a lexicon resulting from the varying nature of clinical text includes misspellings, abbreviations, and little-known synonyms. Discovering additional representations of these terms can increase an NLP system's coverage and speed up the development process.

This module is meant to help discover new lexical variants of any concept. It is not meant to be an automated part of a pipeline, but rather a toolkit to help the system developer come up with an extensive lexicon.

If using this library, please cite:

Chapman AB, Alba PR, Bucher BT, Duvall SL, Patterson OV. **A User-Friendly Interface for Concept Dictionary Expansion Using Word Embeddings and SNOMED-CT**. In: *AMIA Annu Symp Proc*. 2018. 

## Demo 
See LexDiscoverDemo.ipynb for an example walkthrough. 

### Getting Started

**Requirements:**

* [Anaconda's Python distribution](https://www.anaconda.com/download/)
* [PyMedTermino](https://pythonhosted.org/PyMedTermino/tuto_en.html#installation)

**Step 1: Clone the repository**

```
git clone https://github.com/department-of-veterans-affairs/vex.git
cd vex
```

**Step 2: Setup the environment**

```
#Create conda environment
conda create -n vex python=3.6 Anaconda
conda activate vex

#Make compatible with Jupyter Notebooks
conda install ipython
conda install jupyter
conda install nb_conda_kernels 

#Install remaining requirements
pip install -r requirements.txt
```

**Step 3: Run the notebook**

```
jupyter notebook
```

Select `VocabularyExpansion.ipynb` or `VocabularyExpansionDemo.ipynb`. Ensure that the correct kernel is being used.  
