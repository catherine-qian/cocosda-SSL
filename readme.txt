This repositiory provides the source code for joint sound event localization and classification. The acoustic inputs include multi-channel GCCPHAT features and MFCC features.


To run the source code, please follow the next steps:
step 1.
clone this repository to your local directory

step 2. 
Download the reconstructed features here: https://drive.google.com/drive/folders/1NOnVSljflo7_0lkx_XFhWaeNOfhRLUrT?usp=sharing, including
(1) feat618dim.mat: small feature set for debug 
(2) featall.mat: all features 
in the main directory

The features are of dimension (# Segment)*(# Frame)*618, where 
618 = 6 (6 pairs) * 51 (GCCPHAT coefficients) + 1(1microphone) *8 frames* 39 (MFCC coefficients)

step 3. 
run:
python cocosdassl.py -input all -wts0 0 

Notes:
-input: (1)small - use only the small feature set (e.g. feat618dim.mat) for debug purpose (2) all - use all features (e.g. featall.mat) 
-wts0:  weight of the localization task while (1-wts0) indicates the weight of event classification
        in particular, wts0=1 -> event localization only task, wts=0 -> event classification only task
        (for the results reported in the COCOSDA2021 paper, we choose wts0=0.99)
        
        
Note: if you use this source code, please cite the paper:
https://arxiv.org/pdf/2108.02539.pdf

@article{qian2021sloclas,
  title={SLoClas: A Database for Joint Sound Localization and Classification},
  author={Qian, Xinyuan and Sharma, Bidisha and Abridi, Amine El and Li, Haizhou},
  journal={arXiv preprint arXiv:2108.02539},
  year={2021}
}




