# MoM (Minimization of Majorization)

## Description
The main concept of this problem is presented in detail on [[1]](https://www.researchgate.net/publication/309008892_Assisted_Dictionary_Learning_for_fMRI_Data_Analysis). Concisely, in this paper is
presented a new method for the analysis of fMRI data sets, that is capable to incorporate a priori available information, via an efficient
optimization framework.<br/>
In the brain, a number of different functions/processes take place simultaneously; thus, the obtained data consists of a mixture of
various activation signals referred to as sources. The aim of fMRI data analysis is to unmix those sources in order to reveal both their activation patterns as well as the corresponding activated brain areas, associated with each one of the sources.<br/>
From a mathematical point of view, the source unmixing task can be described as a problem of factorization of the data matrix, 
_X_= _D_*_S_

## Pseudocode
![image](https://github.com/patschris/MoM/blob/master/MomPseudocode.png)

## CUDA

## Matlab

## References
[1] https://www.researchgate.net/publication/309008892_Assisted_Dictionary_Learning_for_fMRI_Data_Analysis<br/>
[2] https://github.com/MorCTI/Attom-Assisted-DL
