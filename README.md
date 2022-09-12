## VEx

If using this library, please cite:

Chapman AB, Alba PR, Bucher BT, Duvall SL, Patterson OV. **A User-Friendly Interface for Concept Dictionary Expansion Using Word Embeddings and SNOMED-CT**. In: *AMIA Annu Symp Proc*. 2018. 

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
