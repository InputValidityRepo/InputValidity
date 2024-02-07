# Enhancing Valid Test Input Generation with Distribution Awareness for Deep Neural Networks
This repository contains experiments conducted in the paper 'Enhancing Valid Test Input Generation with Distribution Awareness for Deep Neural Networks'


**Abstract:** Comprehensive testing is significant in improving
the reliability of Deep Learning (DL)-based systems. A plethora
of Test Input Generators (TIGs) has been proposed to generate
test inputs that induce model misbehavior. However, the lack
of validity checking in TIGs often results in the generation of
invalid inputs (i.e., out of the learned distribution), leading to
unreliable testing. To save the effort of manually checking the
validity and improving test efficiency, it is important to assess the
effectiveness of automated validators and identify test selection
metrics that capture data distribution shifts.
In this paper, we validate and improve the testing framework
by incorporating distribution awareness. For validation, we
conduct an empirical study to assess the trustworthiness of four
automated Input Validators (IVs). Our findings revealed that the
accuracy of IVs (agreement with humans) ranged from 49% ∼
77%. Distance-based IVs generally outperform reconstruction-
based and density-based IVs for both classification and regression
tasks. Additionally, we analyze six test selection metrics achieved
by valid and invalid inputs, respectively. The results reveal that
invalid inputs can consistently inflate uncertainty-based metrics.
For improvement, we enhance the existing testing framework
by taking into account valid data distribution through joint
optimization. The results have demonstrated a 2% ∼ 10%
increase in the number of valid inputs by human assessment.

# Requirements
1. Python 3.6
2. Tensorflow 2.2
3. Keras 2.3.1
4. skimage 0.17.2
5. Torch 1.12.1
6. Torchvision 0.13.1
7. Scipy 1.2.1

# Demo
We provide a cmd.txt file in each folder to demonstrate the usage of each technique. Please refer to the relative file for detailed usage.

For example, to run DAIV input validator on MNIST generated test inputs:
```
cd IV/DAIV
python3 IV_MNIST.py
```
To run DAIV input validator on Udacity generated test inputs:
```
python3 IV_driving.py
```

# Data 
1. MNIST: we provide mnist.pkl.gz file that contains the MNIST dataset. Users can also download it from other source.
2. Udacity: Please refer to [UdacityReader](https://github.com/rwightman/udacity-driving-reader) for download and read.


