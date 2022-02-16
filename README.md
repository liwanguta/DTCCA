# Deep Tensor CCA for Multi-view Learning

This is the pytorch implementation of the deep tensor CCA for multi-view learning. 
The paper can be found in https://ieeexplore.ieee.org/abstract/document/9428614. 

## Configuration
This code is tested with Python 3.7 and the dependent python packages can be installed using
```
pip install -r requirements.txt
``` 

## Example run
A sample run on multiple feature dataset can be executed as
```
python dtcca.py 
```
All related parameter setup can be found in the sample script.

## Citation
If you use this code, please kindly cite
```
@article{wong2021deep,
  title={Deep tensor CCA for multi-view learning},
  author={Wong, Hok Shing and Wang, Li and Chan, Raymond and Zeng, Tieyong},
  journal={IEEE Transactions on Big Data},
  year={2021},
  publisher={IEEE}
}
```