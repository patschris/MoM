# MoM (Minimization of Majorization)

## Description
The main concept of this problem is presented in detail on [[1]](https://www.researchgate.net/publication/309008892_Assisted_Dictionary_Learning_for_fMRI_Data_Analysis). Concisely, in this paper is
presented a new method for the analysis of fMRI data sets, that is capable to incorporate a priori available information, via an efficient optimization framework.<br/>

In the brain, a number of different functions/processes take place simultaneously; thus, the obtained data consists of a mixture of
various activation signals referred to as sources. The aim of fMRI data analysis is to unmix those sources in order to reveal both their activation patterns as well as the corresponding activated brain areas, associated with each one of the sources.<br/>

From a mathematical point of view, the source unmixing task can be described as a problem of factorization of the data matrix, 
_X_ ≈ _D_·_S_ where _D_ ∈ ℝ<sup>T×K</sup> is a matrix, whose columns represent the activation patterns or time courses, associated with 
each one of the sources, ℝ<sup>K×N</sup> is the matrix whose rows model the brain areas, activated by the corresponding sources, and
K is the number of sources, whose value is set by the user. The rows of the matrix S are usually referred to as spatial maps.<br/>

Recently, a new method, called Supervised Dictionary Learning (SDL), was presented, leading to enhanced results. The starting point in 
the formulation of the SDL, lies in the splitting of the main dictionary in two parts: _D_ = [**Δ**,**D<sub>F</sub>**] ∈ ℝ<sup>T×K</sup>
where the first part, **Δ**∈ ℝ<sup>T×M</sup>, is constrained to contain the imposed task-related time courses and is considered fixed. 
In contrast, the second part, **D<sub>F</sub>** ∈ ℝ<sup>K×N</sup>, is the variable one to be estimated via DL optimizing arguments.This 
approach provides a more relaxed way of incorporating the a-priori adopted forms of the time courses.<br/>

In the paper [[1]](https://www.researchgate.net/publication/309008892_Assisted_Dictionary_Learning_for_fMRI_Data_Analysis), a new 
Dictionary Learning method is proposed to compute a good estimation of _D_ and _S_, based on Supervised Dictionary Learning (SDL). The 
main idea is to consider that the atoms of the constrained part are not necessarily equal to the a-priori selected ones; instead, a 
looser constraint is employed, embedded in the optimization process. Thus, the strong equality demand is relaxed by a looser similarity 
distance-measuring norm constraint.The starting point is, again, to split the dictionary: _D_ = [**D<sub>C</sub>**,**D<sub>F</sub>**] ∈ 
ℝ<sup>T×K</sup>. In contrast to the SDL approach, however, the constrained part, **D<sub>C</sub>** ∈ ℝ<sup>T×M</sup>, is not considered
fixed any more; instead, it can vary in line with the constrained optimization cost. The simplified optimization task, adopted [here](https://www.researchgate.net/publication/309008892_Assisted_Dictionary_Learning_for_fMRI_Data_Analysis) is:
(**Ď**,**Š**)=argmin<sub>D,S</sub>‖**X**-**DS**‖<sub>F</sub><sup>2</sup> <br/>

As part of our thesis, we manage to accelarate this algorithm using GPU and parallel computing. 

## Code
The code proposed from this paper is available [here](https://github.com/MorCTI/Attom-Assisted-DL), written in matlab. You can see the 
pseudocode in the image below.

![pseudocode](https://github.com/patschris/MoM/blob/master/MomPseudocode.png)

The goal is to rewrite the code using the GPU in order to improve the performance of the given code. We wrote two version of this code: 
one written in CUDA and another written in Matlab using [GPU arrays](https://www.mathworks.com/help/distcomp/gpuarray.html).

## Specs
![specs](https://github.com/patschris/MoM/blob/master/specs.png)

## References
[1] https://www.researchgate.net/publication/309008892_Assisted_Dictionary_Learning_for_fMRI_Data_Analysis<br/>
[2] https://github.com/MorCTI/Attom-Assisted-DL
[3] http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf
