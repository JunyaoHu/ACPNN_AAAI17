# ACPNN_AAAI17

This is an implementation repository of ACPNN/BCPNN (in development).

**[AAAI17]** **ACPNN**: Learning Visual Sentiment Distributions via Augmented Conditional Probability Neural Network

## Installation

```bash
conda create -n ACPNN python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

git clone https://github.com/JunyaoHu/ACPNN_AAAI17
cd ACPNN_AAAI17
pip install -e .
```

## Citation

```
@inproceedings{yang2017learning,
  title={Learning visual sentiment distributions via augmented conditional probability neural network},
  author={Yang, Jufeng and Sun, Ming and Sun, Xiaoxiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={31},
  number={1},
  year={2017}
}
```