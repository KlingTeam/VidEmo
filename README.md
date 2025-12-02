<div align=center>
    <img src="assets/logo.png" width=15%>
    <h1>VidEmo: Affective-Tree Reasoning for Emotion-Centric Video Foundation Models</h1>

<div class="is-size-5 publication-authors">
<span class="author-block">
    <a href="https://zzcheng.top/" target="_blank">Zhicheng Zhang</a><sup>1,†</sup>,
</span>
<span class="author-block">
    Weicheng Wang<sup>1</sup>,
</span>
<span class="author-block">
    <a href="https://yongjie-zhu.github.io/" target="_blank">Yongjie Zhu</a><sup>3,‡</sup>,
</span>
<span class="author-block">
    Wenyu Qin<sup>3</sup>,
</span>
<span class="author-block">
    <a href="https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en/" target="_blank">Pengfei Wan</a><sup>3</sup>,
</span>
<span class="author-block">
    Di Zhang<sup>3</sup>,
</span>
<span class="author-block">
    <a href="https://cv.nankai.edu.cn/" target="_blank">Jufeng Yang</a><sup>1,2,✉</sup>
</span>
</div>

<!-- Institution -->
<div class="is-size-5 publication-authors">
    <sup>1</sup><span class="author-block">Nankai University</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <sup>2</sup><span class="author-block">Pengcheng Laboratory</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <sup>3</sup><span class="author-block">Kuaishou Technology</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>
<div class="is-size-5 publication-authors">
    <sup>†</sup><span class="author-block">Work done at KlingAI</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <sup>‡</sup><span class="author-block">Project Leader</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <sup>✉</sup><span class="author-block">Corresponding Author</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>


**🎉 Accepted by [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/115267) 🎉**



<a href="https://arxiv.org/abs/2511.02712" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Kling--VidEmo-red?logo=arxiv" height="25" />
</a>
<a href="https://zzcheng.top/VidEmo" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-Homepage-blue.svg" height="25" />
</a>
<a href="https://github.com/KlingTeam/VidEmo" target="_blank">
    <img alt="Github" src="https://img.shields.io/badge/⚒️_Github-Code-white.svg" height="25" />
</a>
<a href="https://zzcheng.top/assets/pdf/2025_NeurIPS_VidEmo_poster.pdf" target="_blank">
    <img alt="HF Dataset: Emo-CFG 2.1M" src="https://img.shields.io/badge/📅-Poster-gree.svg" height="25" />
</a>
<br>
<a href="https://huggingface.co/KlingTeam/VidEmo-3B" target="_blank">
    <img alt="HF Model: VidEmo Family" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Kling--VidEmo--3B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/KlingTeam/VidEmo-7B" target="_blank">
    <img alt="HF Model: VidEmo Family" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Kling--VidEmo--7B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/KlingTeam/Emo-CFG" target="_blank">
    <img alt="HF Dataset: Emo-CFG 2.1M" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-Emo--CFG--2.1M-ffc107?color=ffc107&logoColor=white" height="25" />
</a>




<img src="assets/Teaser3.png" width=800 />
</div>

> **TL;DR**: We present an emotion-centric video foundation model trained with fine-grained captions and rationales via affective-tree reasoning guidance, achieving high-level emotional intelligence for video understanding.

## 📈 1. News

- 🔥2025-12-01: Pre-computed results, inference, and evaluation code released.
- 🔥2025-12-01: Creating repository.
- 2025-09-18: Kling-VidEmo has been accepted to NeurIPS 2025！

## ⚒️ 2. Environment Setup

```
conda create -n VidEmo python=3.9
conda activate VidEmo
python -m pip install -r requirements.txt
cd ms-swift
python -m pip install -e .
```

## 💾 3. Emo-CFG Datasets

### 🔐 3.1 Overview of dataset

<img src="assets/datastats.png" width=800 />
</div>

In (a), the data taxonomy organizes the dataset into three primary face perception tasks: Emotion Intelligence, Expression Analysis, and Attribution Perception, covering a wide range of facial features and emotional attributes. (b) The data distribution plots show the relative face area and video duration across different datasets, illustrating the diversity and variety of video data present in Emo-CFG. (c) The annotation distribution includes the breakdown of facial views (head, half, full) and video length, accompanied by a word cloud highlighting the most frequently annotated terms, such as “neutral”, “face”, and “expression”. (d) Data statistics compares Emo-CFG with other emotion and video datasets, showing that Emo-CFG provides a richer set of annotations and label types, including fine-grained emotion, rationales, and comprehensive video data, making it a unique and valuable resource for emotion-centric research.

The `dataset` folder should be structured as follow:

~~~~
Emo-CFG
├── jsons
│   ├── curation
│   │   ├── concat_receipt.py
│   │   ├── v1
│   │   │   └── source.txt
│   │   ├── v2
│   │   │   └── source.txt
│   │   ├── v3
│   │   │   └── source.txt
│   │   ├── v4
│   │   │   └── source.txt
│   │   └── v5
│   ├── test
│   │   ├── attribute
│   │   │   ├── full
│   │   │   └── sampled
│   │   ├── caption
│   │   │   ├── full
│   │   │   └── sampled
│   │   ├── emotion
│   │   │   ├── full
│   │   │   └── sampled
│   │   └── qa
│   │       ├── full
│   │       └── sampled
│   └── train
│       ├── attribute
│       │   ├── full
│       │   └── sampled
│       ├── caption
│       │   ├── full
│       │   └── sampled
│       ├── emotion
│       │   ├── full
│       │   └── sampled
│       ├── qa
│       │   ├── full
│       │   └── sampled
│       └── rationale
│           ├── full
│           └── sampled
└── videos
    ├── AFEW
    ├── AffWild2
    ├── CAER
    ├── CASME
    ├── CAS(ME)2
    ├── CASME2
    ├── CelebV-HQ
    ├── CelebV-Text
    ├── Dfew
    ├── FERV39K
    ├── MAFW
    ├── MEAD
    ├── MELD
    ├── Mer2023
    ├── MOSEI
    ├── MOSI
    ├── PERR
    ├── RAVDESS
    └── SIMS
~~~~

### 🔐 3.2 Access & License

To access the dataset, you must upload a signed End User License Agreement (EULA) via our HuggingFace repository:

[👉 Emo-CFG on HuggingFace](https://huggingface.co/datasets/KlingTeam/Emo-CFG)

> **⚠️ Note**: The copyright of the videos remains with the original owners.
> If you find this work useful, please consider **cite our paper** and **acknowledging the related dataset resources** kindly.






## 🔬 4. VidEmo Family


### 🧊 4.1 Model Collection

To use the model weights, download them from Hugging Face:
- [VidEmo-3B](https://huggingface.co/KlingTeam/VidEmo)
- [VidEmo-7B](https://huggingface.co/KlingTeam/VidEmo)

### 🔮 4.2 Train

##### 🧱 SFT Stage

TBD

##### 🧱 RL Stage

TBD

### 🔮 4.3 Inference

#### 📜 Scripts
Run the following command to perform inference. 
> **Note:** Ensure that the path variables (e.g., `${BASE_DATASET_DIR}`) are defined or replaced with your actual file paths before running.

```bash
VIDEO_MAX_PIXELS=100352 FPS_MAX_FRAMES=16 CUDA_VISIBLE_DEVICES=0 swift infer \
    --val_dataset "${BASE_DATASET_DIR}/${TESTING_DATASET_NAME}" \
    --ckpt_dir "${BASE_CKPT_DIR}/${TESTING_MODEL_NAME}" \
    --result_path "${RESULT_PATH}" \
    --infer_backend vllm \
    --gpu_memory_utilization 0.85 \
    --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --streaming False \
    --max_batch_size 4 \
    --attn_impl flash_attn \
    --limit_mm_per_prompt '{"image": 0, "video": 1}' \
    --max_model_len 49152
```

For a complete batch processing script, please refer to `scripts/inference.sh`

##### 📊 Pre-computed VidEmo and SOTA Results

To facilitate fair comparison and ensure alignment with our reported metrics, we provide the original inference outputs used in our paper. Please refer to the `resutls` folder.

> **Note on Evaluation**: You may use your own GPT version/API key for evaluation. We have observed that while absolute scores may vary for open-form QA data, **the relative ranking of the results remains consistent** across different GPT versions.

### 🔮 4.4 Evaluation

##### Demonstration

```
eval
├─ config.py # GPT configuration
├─ eval_results.py # Evaluation scripts
├─ generate_table.py # CSV & table generator
└─ util.py # Utility functions
```

1. Modify the LLM configuration in `config.py`

   - Modify `API_KEY` to your API key
   - Modify `BASE_URL` to your LLM's base URL
   - We recommend setting `MODEL_NAME` to `gpt-4o-2024-08-06` to better align with the reported results.

2. Execute the evaluation scripts

   ```sh
   python -m eval.eval_results \
   	--input_dir "Path/to/the/input/directory" \
   	--method "method name, e.g. models--Qwen--Qwen2.5-VL-7B-Instruct" \
   	--output_dir "Path/to/the/output/txt/directory" \
   	--retry 50 \ # Optional, maximum retry number
   	--max_concurrency  # Optional, maximum concurrent requests
   ```

   By default, this script will evaluate all tasks defined in `config.py Class Tasks`. You may find example usage for evaluating a specific task in `eval_results.py` line 348.

3. Export the results to CSV files and generate tables

   ```sh
   python -m eval.generate_table \
   	--input_dir "Path/to/where/all/txt/files/stay" \
   	--csv_file_dir "Path/to/the/target/directory/of/csv/file" \ # Optional, default to "input_dir"
   	--table_file_dir "Path/to/the/target/directory/of/table/file"
   ```

   This will generate an `output.csv` CSV file under `csv_file_dir` and a `table.txt` file under `table_file_dir`.

##### Notes

1. The QA evaluation relies on the ground truth annotation file. This is defined in `config.py` under `Tasks.QA.gt_file`. Please also modify this path for a successful evaluation.
2. To customize your own evaluation task, please add another instance of `EvalTask` under the `Tasks` class located in the `config.py` file.


## ⭐ 5. Star History

[![Star History Chart](https://api.star-history.com/svg?repos=KwaiVGI/VidEmo&type=Date)](https://star-history.com/#KwaiVGI/VidEmo&Date)

## 📫 6. Contact

If you have any questions, please feel free to contact:

- Zhicheng Zhang: gloryzzc6@sina.com
- Weicheng Wang: 1394828098wwc@gmail.com

## 🏷️ 7. Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{zhang2025VidEmo,
  author = {Zhang, Zhicheng and Wang, Weicheng and Zhu, Yongjie and Qin, Wenyu and Wan, Pengfei and Zhang, Di and Yang, Jufeng},
  title = {VidEmo: Affective-Tree Reasoning for Emotion-Centric Video Foundation Models},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025}
}
```

## 🥰 8. Acknowledgements

This project stands on the shoulders of giants. We deeply appreciate the [ms-swift](https://github.com/modelscope/ms-swift) library for their excellent codebase. Our dataset is constructed based on the following foundational resources in affective computing. We sincerely thank the authors of these datasets:

| | | | |
| :--- | :--- | :--- | :--- |
| **AFEW** | **AffWild2** | **CAER** | **CASME** |
| **CAS(ME)²** | **CASME2** | **CelebV-HQ** | **CelebV-Text** |
| **DFEW** | **FERV39K** | **MAFW** | **MEAD** |
| **MELD** | **MER2023** | **MOSEI** | **MOSI** |
| **PERR** | **RAVDESS** | **SIMS** | |