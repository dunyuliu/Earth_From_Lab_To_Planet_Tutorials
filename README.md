# UT_Earth_Dynamics_Numerical_Tutorials
This repository hosts the Jupyter notebooks and MATLAB scripts of numerical toturials for UT GEOL 315L: Earth From Lab to Planet, Spring 2023. <br/>

This README includes the following sections: 1. Installation of computing environments; 2. Resources for Jupyter Notebook beginners.

## 1. Installation of Computing Environments
The repository contains Jupyter Notebooks, which use Python and relevant packages, and MATLAB scripts, which use [MATLAB](https://www.mathworks.com/products/matlab.html). Therefore, we need to install a Python environment and MATLAB to use corresponding scripts.

To use Python-based Jupyter Notebooks, there are a few options: <br/>
(a) Generally a Linux system supports Python. <br/> 
(b) Anaconda is a platform good for all operating systems including Windows, Linux and MacOS. An installation guide of Anaconda is provided in section 1.1. <br/>
(c) [Binder](https://mybinder.org/) is a good option to run Jupyter Notebooks on a web browser regardless of operating systems. <br/>
After you open https://mybinder.org, insert the link to our GitHub repository https://github.com/dunyuliu/UT_Earth_Dynamics_Numerical_Tutorials.git to the block under "GitHub repository name or URL". <br/> 
Then, click "launch". Binder will create a docker container. <br/>
It will take a few minutes to build and launch. Then, Jupyter Notebooks will be open in a new web browser tab. <br/>
After you successfully launch your repository on binder, you can copy and share the link in the block under "Copy the URL below and share your Binder with others:" to allow others to use your repository built on binder. <br/> 

Currently, an already built and working repository could be accessed via this [LINK](https://mybinder.org/v2/gh/dunyuliu/UT_Earth_Dynamics_Numerical_Tutorials.git/cf2519046f888db82737bae9977ceb7fc1b7b066) (will be updated in the future). <br/>
(d) Docker (in progress). <br/>

### 1.1 Installation of Anaconda
The link to download Anaconda can be found [here](https://www.anaconda.com/products/distribution). <br/>
(i)  For Windows users, please download the installer and install it as a usual App. 
(ii) For Linux (or Ubuntu subsystem on Windows) users,
#### Download and install
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```
#### Initiate conda with different shells

Right after the installation of Anaconda, you will be asked to initiate conda with the command
```
conda init
```
Or, you can use 
```
conda init --all
```
to initiate conda for all the shells (bash, tcsh, fish, xonsh, zsh, powershell, etc.. NOTE, the command will modify .bashrc and/or .tcshrc). 

#### Alternatives
You may want to initiate conda with the following lines
```
source $Anaconda_root_path"/etc/profile.d/conda.csh"
```
for tcsh or, 
```
source $Anaconda_root_path"etc/profile.d/conda.sh"
```
for bash shell.

## 2. Resources for Jupyter Notebook beginners.
If you are new to Jupyter Notebook, here are some nice resources for beginners. <br/>
[Link1](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) is a Jupyter Notebook beginner's guide that basically explains everything about Jupyter Notebook. <br/>
[Link2](https://www.youtube.com/watch?v=2WL-XTl2QYI) is a 15 min YouTube tutorial that will walk you through Jupyter Notebook installation with Anaconda Python distribution and create and use your first Jupyter Notebook. <br/>  
