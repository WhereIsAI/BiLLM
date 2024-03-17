# BiLLM
Toolkit to convert LLMs from uni- to bi-directional for language understanding tasks such as classification and sentence embeddings. Compatible with 🤗 transformers.

## Usage

1) specify start index for bi-directional layers via `export BiLLM_START_INDEX={layer_index}`. if not specified, default is 0, i.e., all layers are bi-directional. If set to -1, BiLLM is disabled.

2) import LLMs from BiLLM and initialize them as usual.

```diff
- from transformers import LLaMAForSequenceClassification, LLaMATokenizer
+ from billm import (
+    LLamaModel,
+    LLamaForTokenClassification,
+    LLamaForSequenceClassification,
+    LLamaTokenizer,
+ )
```


## Citation

If you use this toolkit in your work, please cite the following paper:

1) For sentence embeddings modeling:

```
```

2) For other tasks:

```bibtex
@article{li2023label,
  title={Label supervised llama finetuning},
  author={Li, Zongxi and Li, Xianming and Liu, Yuzhang and Xie, Haoran and Li, Jing and Wang, Fu-lee and Li, Qing and Zhong, Xiaoqin},
  journal={arXiv preprint arXiv:2310.01208},
  year={2023}
}
```