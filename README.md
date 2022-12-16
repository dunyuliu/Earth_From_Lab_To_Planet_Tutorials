# UT_Earth_Dynamics_Numerical_Tutorials
This repository hosts the Jupyter notebooks of numerical toturials for UT GEOL 371T: Earth Dynamics: Lab to Planet

## 1. Installation of computing environments
The repository contains Jupyter Notebooks, which use python and relevant packages, and MATLAB scripts, which use MATLAB. Therefore, we need to install a python environment and MATLAB to use corresponding scripts.

To use Jupyter Notebooks, there are a few options: 
1. Generally a Linux system supports python. <br/> 
2. Anaconda is a platform good for all operating systems including Windows, Linux ( and MacOS?). An installation guide of Anaconda is provided below. <br/>
3. [Binder](mybinder.org) is a good option to run Jupyter Notebooks on a web browser regardless of operating systems. Binder will create a docker container if we give it the link to our repository. It will take a few minutes to build and launch. <br/>
4. Docker (in progress). <br/>

### 1.1 Installation of Anaconda
The link to download Anaconda can be found [here](https://www.anaconda.com/products/distribution). <br/>
1. For Windows users, please download the installer and install it as a usual App. 
2. For Linux (or Ubuntu subsystem on Windows) users,
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
