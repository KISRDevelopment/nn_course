# Introduction to Modern Machine Learning with Neural Networks 

_Dr. Mohammad M. Khajah
<br>
Associate Research Scientist
<br>
Systems and Software Development 
Department (SSDD)_


This repository contains all the materials for my KISR course on neural networks.

## 1. Installing Python

We will use `Miniconda`, a free minimal installer of Python and its related packages.

### Windows
Download and run the [Miniconda 3 Windows 64-bit installer](https://docs.conda.io/en/latest/miniconda.html). Don't change the default options in the installer.

### Mac OS
Follow the instructions to install [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html) on Mac OS.

**Note**: Make sure to download the appropriate installer for your Mac architecture. For new M1 Macs, pick `Miniconda3 macOS Apple M1 ARM 64-bit bash`. For older Intel-based Macs, pick `Miniconda3 macOS 64-bit bash`.

## 2. Configuring the Environment

1. In Windows, launch the `Anaconda Prompt (miniconda3)` application (open the start menu and just start typing the application name until you find it), or `Terminal` in Mac OS.

2. Create a conda environment named `nncourse`.
```bash
conda create -n nncourse
```
A virtual environment lets developers isolate different versions of the same package. Python packages add extra functionality, such as plotting diagrams, machine learning, linear algebra, web development, etc.

3. Activate the `nncourse` environment.
```bash
conda activate nncourse
```
4. Install the required packages for this course into the `nncourse` environment:
```bash
conda install numpy pandas matplotlib tensorflow
```
`numpy` is the most fundamental library for math, vector, and matrix operations. `pandas` makes it easy to interact with tabular data (csv and excel files). `matplotlib` is used for plotting diagrams. `tensorflow` lets us build neural network models.

5. Install the `jupyterlab` package from the third-party package repository known as condaforge:
```bash
conda install -c conda-forge jupyterlab
```
 `jupyterlab` is an interactive web-browser based layer on top of Python that makes it easy to mix rich documentation and code.

6. You can close the terminal now.

## 3. Running Jupyter Notebook

All coding in this course will be done within the jupyter notebook. To launch it:

1. Launch the `Anaconda Prompt (miniconda3)` application in Windows or `Terminal` in Mac OS.

2. Activate the `nncourse` environment you created earlier.
```bash
conda activate nncourse
```

3. Launch jupyter lab:
```bash
jupyter lab
```
This will launch a local web server running Jupyter and then it will run a web browser and point it to the local server's address. If the browser does not run, copy the URL that appears in the console and paste it into a web browser.

Do *NOT* close the prompt/terminal application while jupyter is running, as you'd kill the local Jupyter server.

4. You can now navigate your local file system via the left side pane in the browser.

## 4. Downloading and Running Course Notebooks

1. On the course's [Github page](https://github.com/KISRDevelopment/nn_course), click the green "Code" button.

2. From the drop down menu, click "Download ZIP".

3. Uncompress the downloaded file into a convenient location (e.g., the desktop).

4. Within the open jupyter lab tab in your web browser, navigate to the location where you uncompressed the file.

5. Double click on one of the `.ipynb` files in the left pane to open it in the right pane.

6. Jupyter notebooks consist of cells. Each cell can be Python code or Markdown (documentation) or plain text. Click on the play button at the top of the notebook to execute the current cell and move to the next one. You can also press `Shift+Enter` to do this. 

## Resources

- Python Numpy Tutorial:
http://cs231n.github.io/python-numpy-tutorial/

- Stanford's CS231N class:
http://cs231n.github.io/
