Steps to run code on Google Colab Pro+:<br><br>
1. Verify the following:<br>
(i)NVIDIA CUDA drivers must be present:<br>
!nvcc --version<br>
Sample output(Output to contain CUDA compiler drive):<br>
nvcc: NVIDIA (R) Cuda compiler driver<br>
Copyright (c) 2005-2023 NVIDIA Corporation<br>
Built on Tue_Aug_15_22:02:13_PDT_2023<br>
Cuda compilation tools, release 12.2, V12.2.140<br>
Build cuda_12.2.r12.2/compiler.33191640_0<br><br>
(ii)Atleast 50GB RAM<br>
!cat /proc/meminfo<br>


2. Install the prelibraries required for Google colab to install Python 3.7.9<br>
!sudo apt-get install libffi-dev<br>

3. Install the Python 3.7.9 version.<br>
Exact version to ensure library compatibility.<br>
Fetch the python files:<br><br>

!wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz<br>

!tar -xvf Python-3.7.9.tgz<br>

%cd Python-3.7.9<br>

Install the libraries for python3.7.9 with the below command<br>
!!chmod 777 ./configure; make clean; make; make install;<br><br>

Verify installation<br>
!python3.7 --version<br>
Output: Python 3.7.9<br><br>

4. Upload all code files and data, pickle files onto Google drive<br>

Dataset files can be downloaded from below link:<br>
https://iowastate-my.sharepoint.com/:f:/g/personal/hritikz_iastate_edu/EtTTsjdBxY1NsyQ-Z0Xr6YUBX3uHv4wg42rzpyCdYrDOEQ?e=Mm3Mgo<br>

Code can be cloned using git clone.<br>
At the top level, create two folders data/ and pickle/ to add the .csv files and the remaining files into pickle/.<br>

5. Run the below command to access the drive files<br><br>

from google.colab import drive<br>
drive.mount('/content/drive')<br>

6. Create virtual env using python 3.7.9 at the top level.<br>
python3.7 -m venv . <br>
Check if python and pip are inside the env using "which pip" and "which python"<br>

7. Run "pip install -r requirements.txt" <br>
8. Run "python -W ignore train.py" to begin training on 12 million rides and evaluation on the next 1 month of dataset.<br>
9. Epochs, loss, Qmax trends are stored in output file "epoch_qmax_loss.txt"<br>
10. Intermediate DQN score outputs are generated for each episode(20 in total)<br>
These files can be used for interpretation.<br><br>
