# BiLLM
Tool for converting LLMs from uni-directional to bi-directional for tasks like classification and sentence embeddings. Compatible with 🤗 transformers.

<a href="https://arxiv.org/abs/2310.01208">
    <img src="https://img.shields.io/badge/Arxiv-2310.01208-yellow.svg?style=flat-square" alt="https://arxiv.org/abs/2310.01208" />
</a>
<a href="https://arxiv.org/abs/2311.05296">
    <img src="https://img.shields.io/badge/Arxiv-2311.05296-yellow.svg?style=flat-square" alt="https://arxiv.org/abs/2311.05296" />
</a>
<a href="https://pypi.org/project/billm/">
    <img src="https://img.shields.io/pypi/v/billm?style=flat-square" alt="PyPI version" />
</a>
<a href="https://pypi.org/project/billm/">
    <img src="https://img.shields.io/pypi/dm/billm?style=flat-square" alt="PyPI Downloads" />
</a>
<a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" />
</a>
<a href="https://pdm-project.org">
    <img src="https://img.shields.io/badge/pdm-managed-blueviolet" alt="https://pdm-project.org" />
</a>


## Usage

1) `python -m pip install -U billm`

2) Specify start index for bi-directional layers via `export BiLLM_START_INDEX={layer_index}`. if not specified, default is 0, i.e., all layers are bi-directional. If set to -1, BiLLM is disabled.

3) Import LLMs from BiLLM and initialize them as usual with transformers.

```diff
- from transformers import (
-    LLamaModel,
-    LLamaForSequenceClassification,
-    MistralModel,
-    MistralForSequenceClassification
- )

+ from billm import (
+    LLamaModel,
+    LLamaForSequenceClassification,
+    LLamaForTokenClassification,
+    MistralModel,
+    MistralForSequenceClassification,
+    MistralForTokenClassification,
+ )
```

## Examples

### NER

```bash
$ cd examples
$ WANDB_MODE=disabled BiLLM_START_INDEX=0 CUDA_VISIBLE_DEVICES=0 python billm_ner.py \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--dataset_name_or_path conll2003
```

## Supported Models

- LLaMA
- Mistral

## Citation

If you use this toolkit in your work, please cite the following paper:

1) For sentence embeddings modeling:

```bibtex
@inproceedings{li2024bellm,
    title = "BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings",
    author = "Li, Xianming and Li, Jing",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
    year = "2024",
    publisher = "Association for Computational Linguistics"
}
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
