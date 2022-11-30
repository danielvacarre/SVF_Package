# SVF Package
![Build Status](./images/build_passing.svg)

## Support Vector Frontiers package
Support Vector Frontier (see [Valero-Carreras et al., 2022]) (SVF) is a machine learning technique, based on Support Vector Machines (SVM), used for the estimation of production functions.

### Contents

- **Different types of SVF:**
  + **Complete Support Vector Frontiers  (SVFC):**
  + **Simplified Support Vector Frontier (SSVF):**
  + **Simplified Support Vector Frontier Dual (SVF dual):**
  + **Support Vector Frontier Splines (SVF-SP):**
- **Cross Validation:**
  + **k-folds:**
  + **Train test split:**
- **Calculation of inefficiencies:**
  + **FDH, DEA, SVF, CSVF:**
    + **BCC input orientation:**
    + **BCC output orientation:**
    + **Weighted aditive:**
    + **Directional distance:**
    + **Russell Input Orientation:**
    + **Rusell Output Orientation:**
    + **Enhanced Russell Orientation:**

### Instalation
For installing the package you can use the command:
> **pip install SVF_Package**

SVF Package uses different packages for correct execution:
+ Pandas
+ Docplex
+ Numpy
+ Scikit-learn

### Warning
SVF is an algorithm with a high computational cost. The performance of the algorithm 
largely depends on the amount of inputs and sample size. 

The code has been tested for a ...

### Examples



### Contact

**Daniel Valero Carreras**: [email]

**[University Miguel Hernández (Elche)]**

**[Institute Center of Operations Research (CIO)]**

Bugs can be reported to [github repository]. The code can also be found there.


### License

**MIT License**

**Copyright (c) 2022- DANIEL VALERO CARRERAS**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


[//]: #

[Valero-Carreras et al., 2022]: <https://www.sciencedirect.com/science/article/pii/S0305054822000600>
[github repository]: <https://github.com/danielvacarre/SVF_Package>
[email]: dvalero@umh.es
[University Miguel Hernández (Elche)]:https://www.umh.es/
[Institute Center of Operations Research (CIO)]: https://cio.umh.es/