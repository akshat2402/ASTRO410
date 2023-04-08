# Homework 2 README file for Levenberg-Marquardt algorithm

**List of files:**
hw2-Chaturvedi.py
LorentzianFit.jpg
GaussianFit.jpg
hw2_fitting.dat
hw-Chaturvedi.pdf

**Instructions**
This program is written, **using my own code**, in python 3.10, and is in 
the form of a pyhton script. To run this file, the user must go to their 
terminal, change their working directory into the folder containing this 
file, as well as the data file titled "hw2_fitting.dat" and then run the 
following command:

python hw2-Chaturvedi.py
 **or**
python3 hw2-Chaturvedi.py

Upon doing this, the script will run, and will produce two .jpg files 
titled LorentzianFit.jpg and GaussianFit.jpg, with the data points and the 
fits drawn on top of them with the error bars on the points.

NOTE: This script takes around 30 seconds to run in total, with both fits. 
I understand that this is significantly longer than is expected. I believe 
that this is due to the _for_ loops that I have in my derivative 
functions. I tried to make the code run faster by moving the derivatives 
outside the functions but that wasn't able to make the code quicker by any 
large metric. I apologize about that. 

