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
