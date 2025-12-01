import argparse
import csv
import sys

import pandas as pd
import numpy as np

from pathlib import Path


TAB1_COLS = [
    ["CelebV-HQ_Appearance-Recognition"],
    ["CelebV-Text_Appearance-Caption"],
    ["CelebV-HQ_Action-Recognition"],
    ["CelebV-Text_Action-Caption"],
    ["MEAD_ID"],
    ["MEAD_Head-Pose"],
    [
        "CelebV-HQ_Appearance-Recognition",
        "CelebV-Text_Appearance-Caption",
        "CelebV-HQ_Action-Recognition",
        "CelebV-Text_Action-Caption",
        "MEAD_ID",
        "MEAD_Head-Pose",
    ],
    ["QA_test_sampling_all_Eyes", "QA_test_sampling_all_Eyebrows"],
    ["QA_test_sampling_all_Mouth"],
    ["QA_test_sampling_all_Nose"],
    ["QA_test_sampling_all_Hair"],
    ["QA_test_sampling_all_Chin"],
    ["QA_test_sampling_all_Face_shape"],
    ["QA_test_sampling_all_Facial_features"],
    ["QA_test_sampling_all_Accessories"],
    ["QA_test_sampling_all_Age"],
    ["QA_test_sampling_all_Gender"],
    ["QA_test_sampling_all_Skin"],
    ["QA_test_sampling_all_Body_actions", "QA_test_sampling_all_Facial_actions", "QA_test_sampling_all_Head_actions"],
    [
        "QA_test_sampling_all_Facial_features",
        "QA_test_sampling_all_Gender",
        "QA_test_sampling_all_Chin",
        "QA_test_sampling_all_Head_actions",
        "QA_test_sampling_all_Eyebrows",
        "QA_test_sampling_all_Nose",
        "QA_test_sampling_all_Mouth",
        "QA_test_sampling_all_Accessories",
        "QA_test_sampling_all_Body_actions",
        "QA_test_sampling_all_Hair",
        "QA_test_sampling_all_Facial_actions",
        "QA_test_sampling_all_Face_shape",
        "QA_test_sampling_all_Skin",
        "QA_test_sampling_all_Eyes",
        "QA_test_sampling_all_Age",
    ],
]

TAB1_FACTORS = [
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [1, 1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1, 1, 1],
    [1, 1, 1, 1 / 3, 1 / 2, 1, 1, 1, 1 / 3, 1, 1 / 3, 1, 1, 1 / 2, 1],
]

TAB2_COLS = [
    ["DFEW_Single-Label-Emotion-Recognition", "MAFW_Single-Label-Emotion-Recognition", "MER2023_Single-Label-Emotion-Recognition"],
    ["MAFW_Multiple-Label-Emotion-Recognition"],
    ["MEAD_Fine-grained-Emotion-Recognition", "RAVDESS_Fine-grained-Emotion-Recognition"],
    [
        "DFEW_Single-Label-Emotion-Recognition",
        "MAFW_Single-Label-Emotion-Recognition",
        "MER2023_Single-Label-Emotion-Recognition",
        "MAFW_Multiple-Label-Emotion-Recognition",
        "MEAD_Fine-grained-Emotion-Recognition",
        "RAVDESS_Fine-grained-Emotion-Recognition",
    ],
    ["MOSEI_Single-Label-Sentiment-Recognition", "MOSI_Single-Label-Sentiment-Recognition"],
    ["CHSIMIv1_Fine-grained-Sentiment-Recognition"],
    ["MOSEI_Single-Label-Sentiment-Recognition", "MOSI_Single-Label-Sentiment-Recognition", "CHSIMIv1_Fine-grained-Sentiment-Recognition"],
    ["CAS_Single-Label-Micro-Expression-Recognition"],
    ["AffWild2_Action_Unit_Detection"],
    ["CAS_Single-Label-Micro-Expression-Recognition", "AffWild2_Action_Unit_Detection"],
    ["CelebV-Text_Emotion-Caption"],
    ["PERR_Single-Label-Conversational-Emotion-Recognition", "MELD_Single-Label-Conversational-Emotion-Recognition_new"],
    [
        "CelebV-Text_Emotion-Caption",
        "PERR_Single-Label-Conversational-Emotion-Recognition",
        "MELD_Single-Label-Conversational-Emotion-Recognition_new",
    ],
]

TAB2_FACTORS = [
    [1, 1, 1],
    [1],
    [1, 1],
    [1 / 3, 1 / 3, 1 / 3, 1, 1 / 2, 1 / 2],
    [1, 1],
    [1],
    [1 / 2, 1 / 2, 1],
    [1],
    [1],
    [1, 1],
    [1],
    [1, 1],
    [1, 1 / 2, 1 / 2],
]

TAB3_COLS = [
    ["caption_test_sampling_v2_all_example_lab"],
    ["caption_test_sampling_v2_all_example_clu"],
    ["caption_test_sampling_v2_all_example_ia"],
    ["caption_test_sampling_v2_all_example_ra"],
    ["caption_test_sampling_v2_all_example_vtr"],
    ["caption_test_sampling_v2_all_example_flu"],
    [
        "caption_test_sampling_v2_all_example_lab",
        "caption_test_sampling_v2_all_example_clu",
        "caption_test_sampling_v2_all_example_ia",
        "caption_test_sampling_v2_all_example_ra",
        "caption_test_sampling_v2_all_example_vtr",
        "caption_test_sampling_v2_all_example_flu",
    ],
]

TAB3_FACTORS = [
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [
        1,
        1,
        1,
        1,
        1,
        1,
    ],
]


class TableGenerator:
    SOTAs = [
        # "models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf",
        # "models--OpenGVLab--InternVL2_5-2B",
        # "models--DAMO-NLP-SG--VideoLLaMA3-2B",
        # "models--mPLUG--mPLUG-Owl3-2B-241014",
        # "models--Qwen--Qwen2.5-VL-3B-Instruct",
        # "sharegpt4video_8b",
        # "models--OpenGVLab--InternVL2_5-8B",
        # "models--llava-hf--LLaVA-NeXT-Video-7B-hf",
        # "models--llava-hf--llava-onevision-qwen2-7b-ov-hf",
        # "models--DAMO-NLP-SG--VideoLLaMA3-7B",
        # "models--Isotr0py--LLaVA-Video-7B-Qwen2-hf",
        "models--mPLUG--mPLUG-Owl3-7B-240728",
        # "models--Qwen--Qwen2.5-VL-7B-Instruct",
    ]

    PREFIXES = [
        # "LLaVA-OV~\cite{li2024llava}\n&1B&",
        # "InternVL2.5~\cite{chen2024expanding}\n&2B&",
        # "VideoLLaMA3~\cite{zhang2025videollama}\n&2B&",
        # "mPLUG-Owl3~\cite{ye2024mplug}\n&2B&",
        # "Qwen2.5-VL~\cite{qwen2025qwen}\n&3B&",
        # "ShareGPT4Video~\cite{chen2024sharegpt4video}\n&8B&",
        # "InternVL2.5~\cite{chen2024expanding}\n&8B&",
        # "LLaVA-N-Video~\cite{liu2024llavanext}\n&7B&",
        # "LLaVA-OV~\cite{li2024llava}\n&7B&",
        # "VideoLLaMA3~\cite{zhang2025videollama}\n&7B&",
        # "LLaVA-Video~\cite{zhang2024video}\n&7B&",
        "mPLUG-Owl3~\cite{ye2024mplug}\n&7B&",
        # "Qwen2.5-VL~\cite{qwen2025qwen}\n&7B&",
        "\midrule\n\\rowcolor{LightCyan}\n\model\n&7B&",
    ]

    REJS = [
        "VidEmoEmo-CFG_bs-512_data-ATTR_OPEN_EMO_500k_lr-4e-5v7-20250509-021614checkpoint-1400",
        "VidEmoEmo-CFG_bs-512_data-OPEN_150k_lr-4e-5v146-20250505-031643checkpoint-550",
    ]

    def __init__(self, input_dir: str, csv_file_dir: str, table_file_dir: str):
        self.input_dir = Path(input_dir)
        self.csv_file = Path(csv_file_dir) / "output.csv"
        self.table_file = Path(table_file_dir) / "table.txt"

    def make_csv(self):
        aggregated_data = []

        # Iterate over the entire input directory and
        # search for all .txt files
        for txt_file in self.input_dir.rglob("*.txt"):
            dataset_name = txt_file.stem  # filename as the dataset name

            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        print(f"Warning: incorrect line {line} found in file {txt_file}")
                        continue

                    score, model_name, test_type = parts
                    try:
                        score = float(score)
                    except ValueError:
                        print(f"Warning: cannot convert to float number. In file {txt_file}, line {line}")
                        continue

                    aggregated_data.append({"dataset": dataset_name, "model": model_name, "score": score, "test_type": test_type})

        with open(self.csv_file, "w+", newline="", encoding="utf-8-sig") as csv_file:
            fieldnames = ["dataset", "model", "score", "test_type"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for data in aggregated_data:
                writer.writerow(data)

    def generate_table(self):
        df = pd.read_csv(self.csv_file)

        with open(self.table_file, "w+") as f:
            ori = sys.stdout
            sys.stdout = f  # So that we can ues print to write content to files

            ours = sorted([i for i in df["model"].unique().tolist() if i not in self.SOTAs and i not in self.REJS])

            for i in range(len(ours)):
                print("-------------------------" + ours[i] + "-----------------------------")
                print("-------------------------  Table 1  -----------------------------")
                variant = ours[i : i + 1]
                methods = self.SOTAs + variant

                table1_scores = np.ones((len(methods), len(TAB1_COLS))) * -1.0
                for method_idx, method in enumerate(methods):
                    for column_idx, column_datasets in enumerate(TAB1_COLS):
                        score = []
                        for column_dataset_idx, column_dataset in enumerate(column_datasets):
                            factor = TAB1_FACTORS[column_idx][column_dataset_idx]
                            # Filter the corresponding line
                            filtered_df = df[(df["dataset"] == column_dataset) & (df["model"] == method)]
                            # Found, extract the score
                            if not filtered_df.empty:
                                score.append(filtered_df["score"].values[0] * factor)
                            else:
                                score.append(-1e7)  # Not found, default to -1.0
                        avg_score = sum(score) / sum(TAB1_FACTORS[column_idx])
                        if avg_score >= 0:
                            table1_scores[method_idx, column_idx] = avg_score

                # maximum in each column
                max_scores = np.max(table1_scores, axis=0)
                max_count = 0
                for method_idx in range(len(methods)):
                    row = []
                    for column_idx in range(len(TAB1_COLS)):
                        s = table1_scores[method_idx, column_idx]
                        if s >= 0:
                            if s == 100:
                                formatted_score = f"{int(s)}"  # 100 -> integer output
                            else:
                                formatted_score = f"{int(s):02d}.{int((s - int(s)) * 10):1d}"

                            # Current score is the max, append `\textbf{}`
                            if s == max_scores[column_idx]:
                                if method_idx == len(methods) - 1:
                                    max_count += 1
                                row.append(f"\\textbf{{{formatted_score}}}")
                            else:
                                row.append(formatted_score)

                        else:
                            row.append("    ")  # score < 0, append blank spaces
                    # print(prefixs[method_idx]+"&".join(row)+'\\\\')
                print("Best: ", max_count, " / ", len(TAB1_COLS))
                print("Outperform SOTA in each metrics: ", np.mean(table1_scores[-1]) - np.mean(np.max(table1_scores[:-1], axis=0)))
                print("-------------------------  Table 1  -----------------------------")
                print()
                print()
                print("-------------------------  Table 2  -----------------------------")
                variant = ours[i : i + 1]
                methods = self.SOTAs + variant

                table2_scores = np.ones((len(methods), len(TAB2_COLS))) * -1.0
                for method_idx, method in enumerate(methods):
                    for column_idx, column_datasets in enumerate(TAB2_COLS):
                        score = []
                        for column_dataset_idx, column_dataset in enumerate(column_datasets):
                            factor = TAB2_FACTORS[column_idx][column_dataset_idx]
                            # Filter the corresponding line
                            filtered_df = df[(df["dataset"] == column_dataset) & (df["model"] == method)]
                            # Found, extract the score
                            if not filtered_df.empty:
                                score.append(filtered_df["score"].values[0] * factor)
                            else:
                                score.append(-1e7)  # Not found, default to -1.0
                        avg_score = sum(score) / sum(TAB2_FACTORS[column_idx])
                        if avg_score >= 0:
                            table2_scores[method_idx, column_idx] = avg_score

                # maximum in each column
                max_scores = np.max(table2_scores, axis=0)
                max_count = 0
                for method_idx in range(len(methods)):
                    row = []
                    for column_idx in range(len(TAB2_COLS)):
                        s = table2_scores[method_idx, column_idx]
                        if s >= 0:
                            if s == 100:
                                formatted_score = f"{int(s)}"  # 100 -> integer output
                            else:
                                formatted_score = f"{int(s):02d}.{int((s - int(s)) * 10):1d}"

                            # Current score is the max, append `\textbf{}`
                            if s == max_scores[column_idx]:
                                if method_idx == len(methods) - 1:
                                    max_count += 1
                                row.append(f"\\textbf{{{formatted_score}}}")
                            else:
                                row.append(formatted_score)

                        else:
                            row.append("    ")  # score < 0, append blank spaces
                    # print(prefixs[method_idx]+"&".join(row)+'\\\\')
                print("Best: ", max_count, " / ", len(TAB2_COLS))
                print("Outperform SOTA in each metrics: ", np.mean(table2_scores[-1]) - np.mean(np.max(table2_scores[:-1], axis=0)))

                print("-------------------------  Table 2  -----------------------------")
                print()
                print()
                print("-------------------------  Table 3  -----------------------------")
                variant = ours[i : i + 1]
                methods = self.SOTAs + variant

                table3_scores = np.ones((len(methods), len(TAB3_COLS))) * -1.0
                for method_idx, method in enumerate(methods):
                    for column_idx, column_datasets in enumerate(TAB3_COLS):
                        score = []
                        for column_dataset_idx, column_dataset in enumerate(column_datasets):
                            factor = TAB3_FACTORS[column_idx][column_dataset_idx]
                            # Filter the corresponding line
                            filtered_df = df[(df["dataset"] == column_dataset) & (df["model"] == method)]
                            # Found, extract the score
                            if not filtered_df.empty:
                                score.append(filtered_df["score"].values[0] * factor)
                            else:
                                score.append(-1e7)  # Not found, default to -1.0
                        avg_score = sum(score) / sum(TAB3_FACTORS[column_idx])
                        if avg_score >= 0:
                            table3_scores[method_idx, column_idx] = avg_score

                # maximum in each column
                max_scores = np.max(table3_scores, axis=0)
                max_count = 0
                for method_idx in range(len(methods)):
                    row = []
                    for column_idx in range(len(TAB3_COLS)):
                        s = table3_scores[method_idx, column_idx]
                        if s >= 0:
                            if s == 100:
                                formatted_score = f"{int(s)}"  # 100 -> integer output
                            else:
                                formatted_score = f"{int(s):02d}.{int((s - int(s)) * 10):1d}"

                            # Current score is the max, append `\textbf{}`
                            if s == max_scores[column_idx]:
                                if method_idx == len(methods) - 1:
                                    max_count += 1
                                row.append(f"\\textbf{{{formatted_score}}}")
                            else:
                                row.append(formatted_score)

                        else:
                            row.append("    ")  # score < 0, append blank spaces
                    # print(prefixs[method_idx]+"&".join(row)+'\\\\')
                print("Best: ", max_count, " / ", len(TAB3_COLS))
                print("Outperform SOTA in each metrics: ", np.mean(table3_scores[-1]) - np.mean(np.max(table3_scores[:-1], axis=0)))

                print("-------------------------  Table 3  -----------------------------")
                print("-------------------------" + ours[i] + "-----------------------------")
                print()
                print()
                print()
                print()
                print()

            for i in range(len(ours)):
                print("-------------------------" + ours[i] + "-----------------------------")
                print("-------------------------  Table 1  -----------------------------")
                variant = ours[i : i + 1]
                methods = self.SOTAs + variant

                table1_scores = np.ones((len(methods), len(TAB1_COLS))) * -1.0
                for method_idx, method in enumerate(methods):
                    for column_idx, column_datasets in enumerate(TAB1_COLS):
                        score = []
                        for column_dataset_idx, column_dataset in enumerate(column_datasets):
                            factor = TAB1_FACTORS[column_idx][column_dataset_idx]
                            # Filter the corresponding line
                            filtered_df = df[(df["dataset"] == column_dataset) & (df["model"] == method)]
                            # Found, extract the score
                            if not filtered_df.empty:
                                score.append(filtered_df["score"].values[0] * factor)
                            else:
                                score.append(-1e7)  # Not found, default to -1.0
                        avg_score = sum(score) / sum(TAB1_FACTORS[column_idx])
                        if avg_score >= 0:
                            table1_scores[method_idx, column_idx] = avg_score

                # maximum in each column
                max_scores = np.max(table1_scores, axis=0)
                max_count = 0
                for method_idx in range(len(methods)):
                    row = []
                    for column_idx in range(len(TAB1_COLS)):
                        s = table1_scores[method_idx, column_idx]
                        if s >= 0:
                            if s == 100:
                                formatted_score = f"{int(s)}"  # 100 -> integer output
                            else:
                                formatted_score = f"{int(s):02d}.{int((s - int(s)) * 10):1d}"

                            # Current score is the max, append `\textbf{}`
                            if s == max_scores[column_idx]:
                                if method_idx == len(methods) - 1:
                                    max_count += 1
                                row.append(f"\\textbf{{{formatted_score}}}")
                            else:
                                row.append(formatted_score)

                        else:
                            row.append("    ")  # score < 0, append blank spaces
                    print(self.PREFIXES[method_idx] + "&".join(row) + "\\\\")
                print("-------------------------  Table 1  -----------------------------")
                print()
                print()
                print("-------------------------  Table 2  -----------------------------")
                variant = ours[i : i + 1]
                methods = self.SOTAs + variant

                table2_scores = np.ones((len(methods), len(TAB2_COLS))) * -1.0
                for method_idx, method in enumerate(methods):
                    for column_idx, column_datasets in enumerate(TAB2_COLS):
                        score = []
                        for column_dataset_idx, column_dataset in enumerate(column_datasets):
                            factor = TAB2_FACTORS[column_idx][column_dataset_idx]
                            # Filter the corresponding line
                            filtered_df = df[(df["dataset"] == column_dataset) & (df["model"] == method)]
                            # Found, extract the score
                            if not filtered_df.empty:
                                score.append(filtered_df["score"].values[0] * factor)
                            else:
                                score.append(-1e7)  # Not found, default to -1.0
                        avg_score = sum(score) / sum(TAB2_FACTORS[column_idx])
                        if avg_score >= 0:
                            table2_scores[method_idx, column_idx] = avg_score

                # maximum in each column
                max_scores = np.max(table2_scores, axis=0)
                max_count = 0
                for method_idx in range(len(methods)):
                    row = []
                    for column_idx in range(len(TAB2_COLS)):
                        s = table2_scores[method_idx, column_idx]
                        if s >= 0:
                            if s == 100:
                                formatted_score = f"{int(s)}"  # 100 -> integer output
                            else:
                                formatted_score = f"{int(s):02d}.{int((s - int(s)) * 10):1d}"

                            # Current score is the max, append `\textbf{}`
                            if s == max_scores[column_idx]:
                                if method_idx == len(methods) - 1:
                                    max_count += 1
                                row.append(f"\\textbf{{{formatted_score}}}")
                            else:
                                row.append(formatted_score)

                        else:
                            row.append("    ")  # score < 0, append blank spaces
                    print(self.PREFIXES[method_idx] + "&".join(row) + "\\\\")

                print("-------------------------  Table 2  -----------------------------")
                print()
                print()
                print("-------------------------  Table 3  -----------------------------")
                variant = ours[i : i + 1]
                methods = self.SOTAs + variant

                table3_scores = np.ones((len(methods), len(TAB3_COLS))) * -1.0
                for method_idx, method in enumerate(methods):
                    for column_idx, column_datasets in enumerate(TAB3_COLS):
                        score = []
                        for column_dataset_idx, column_dataset in enumerate(column_datasets):
                            factor = TAB3_FACTORS[column_idx][column_dataset_idx]
                            # Filter the corresponding line
                            filtered_df = df[(df["dataset"] == column_dataset) & (df["model"] == method)]
                            # Found, extract the score
                            if not filtered_df.empty:
                                score.append(filtered_df["score"].values[0] * factor)
                            else:
                                score.append(-1e7)  # Not found, default to -1.0
                        avg_score = sum(score) / sum(TAB3_FACTORS[column_idx])
                        if avg_score >= 0:
                            table3_scores[method_idx, column_idx] = avg_score

                # maximum in each column
                max_scores = np.max(table3_scores, axis=0)
                max_count = 0
                for method_idx in range(len(methods)):
                    row = []
                    for column_idx in range(len(TAB3_COLS)):
                        s = table3_scores[method_idx, column_idx]
                        if s >= 0:
                            if s == 100:
                                formatted_score = f"{int(s)}"  # 100 -> integer output
                            else:
                                formatted_score = f"{int(s):02d}.{int((s - int(s)) * 10):1d}"

                            # Current score is the max, append `\textbf{}`
                            if s == max_scores[column_idx]:
                                if method_idx == len(methods) - 1:
                                    max_count += 1
                                row.append(f"\\textbf{{{formatted_score}}}")
                            else:
                                row.append(formatted_score)

                        else:
                            row.append("    ")  # score < 0, append blank spaces
                    print(self.PREFIXES[method_idx] + "&".join(row) + "\\\\")

                print("-------------------------  Table 3  -----------------------------")
                print("-------------------------" + ours[i] + "-----------------------------")
                print()
                print()
                print()
                print()
                print()

            sys.stdout = ori

    def execute(self):
        self.make_csv()
        self.generate_table()

        print(f">>> CSV file generated in {str(self.csv_file.absolute())}")
        print(f">>> Table generated in {str(self.table_file.absolute())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="input directory, where the txt files stay",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--csv_file_dir",
        help="csv file directory, default to input directory",
        type=str,
    )
    parser.add_argument(
        "--table_file_dir",
        help="table file directory",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    csv_file_dir = args.csv_file_dir or input_dir
    table_file_dir = args.table_file_dir

    gen = TableGenerator(input_dir, csv_file_dir, table_file_dir)
    gen.execute()
