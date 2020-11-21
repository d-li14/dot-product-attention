# dot-product attention

|      Architecture     | Top-1 Acc. (%) | Top-5 Acc. (%) | Download |
|:---------------------:|:---------:|:---------:|:--------:|
|       ResNet-50       | 76.74 | 93.47 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ERoYkljPhxZLu29oU2Fqg2kByYxQ3Z57m4CbZtDPqHgTIQ?e=gWxIm8) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ER0CDw-cKD5Amxpq2sl9lYwBiN6rEjtgObPC992Y2rEzCg?e=NfzcOI) |
|       [NL-ResNet-50](https://arxiv.org/abs/1711.07971)       | 76.55 | 92.99 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ETr2LZ9PMwROiaKXJMxWB0YBVgjBEWwryN1QwwN0o1L3eA?e=vwJ47U) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EQBxNlB9v_JNrcOUr3uI92YBYHTdrvvyRq1gz335uwxdVg?e=Ou69Le) |
|       [A^2-ResNet-50](https://arxiv.org/abs/1810.11579)       | 77.24 | 93.66 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EXKheJBpGYlKlphF6mY5d_IBVa8hTI7Jzd7sJ4AeTo1_5w?e=BqR6qP) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EVAb_PE4lPtJsW4rZx_r8nIBUymj-u1oASy6HO4fRwW_9Q?e=6i1dKg) |
|       [GloRe-ResNet-50](https://arxiv.org/abs/1811.12814)       | 77.81 | 93.99 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EUuM6oY9tOFGlmVCx9FFsegBu-OgzETyNfCEhfPa1PrjiQ?e=b9FlSl) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EVlWVUkbbHZCirzjpUeOCSMB4E6_eCw3FeaLo0Oxi-VcAA?e=3jGIQs) |
|       [AA-ResNet-50](https://arxiv.org/abs/1904.09925)<sup>†</sup>       | 77.57 | 93.73 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ES03Xtue75hJoyLVXyT07LABz_TqtagYxmqN1g0k93f3OQ?e=JrjyWX) |

Models are trained on 32 GPUs with the mini-batch size 32 per GPU for 100 epochs. The SGD optimizer with initial learning rate 0.4, momentum 0.9, weight decay 0.0001 is adopted for training. The learning rate anneals following the cosine schedule, with linear warmup for the first 5 epochs with the warmup ratio of 0.25.

† initial learning rate 0.1 w/o warmup for 100 epochs, w/ label smoothing, mini-batch size 32 per GPU on 8 GPUs
