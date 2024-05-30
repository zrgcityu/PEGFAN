Implementation of PEGFAN (Permutation Equivariant Graph Framelet Augmented Network), adpated from [FSGNN](https://github.com/sunilkmaurya/FSGNN) . Please refer to [our paper](https://arxiv.org/abs/2306.04265) for details.

It would be very appreciated if you mention our work or apply our codes and cite
```
@article{li2024permutation,
  title={Permutation equivariant graph framelets for heterophilous graph learning},
  author={Li, Jianfei and Zheng, Ruigang and Feng, Han and Li, Ming and Zhuang, Xiaosheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```

A list of Python packages in our virtual environment is contained in packages.txt.
Other specs: Ubuntu 18.04, CUDA Version 11.0, Graphics Card: NVIDIA RTX 3090

To obtain result on Chameleon (Ours,Type c, h=4) in Table III, run the following command in shell:

bash grid_search_scripts/heterophily_search.sh chameleon

and select test-set result (right column in csv files) according to the best validation set result (left column in csv files).


To obtain other results, modify the hyperparameter in bash scripts.
(use homophily_search.sh for homophilous datasets(cora,citeseer,pubmed) and syn_search.sh for synthetic data, see the scripts for details)

Most of the results in our paper are collected in folder "result_collections", new results after running the codes will be stored in folder "results" and "syn_results".

Framelets are stored in folder "framelets". Delete the files then the framelets will regenerate by running the code.
