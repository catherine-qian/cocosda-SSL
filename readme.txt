This repositiory provides the source code for joint sound event localization and classification. The acoustic inputs include multi-channel GCCPHAT features and MFCC features.

You can download the reconstructed features here:
https://drive.google.com/drive/folders/1NOnVSljflo7_0lkx_XFhWaeNOfhRLUrT?usp=sharing

feat618dim.mat: small feature set for debug
featall.mat: all features 


To run the source code, please follow the next steps:
step 1.
clone this repository to your local directory

step 2. 
Download the reconstructed features here: https://drive.google.com/drive/folders/1NOnVSljflo7_0lkx_XFhWaeNOfhRLUrT?usp=sharing
including 
(1) feat618dim.mat: small feature set for debug 
(2) featall.mat: all features  

The features are of dimension (# Segment)*(# Frame)*618, where 
618 = 6 (6 pairs) * 51 (GCCPHAT coefficients) + 4 (4 microphone) * 78 (MFCC coefficients)


