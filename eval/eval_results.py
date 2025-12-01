import asyncio
import argparse

import json
import logging
import os

from typing import Any
from pathlib import Path
from functools import partial

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from .config import Tasks, EvalTask
from .util import LLMEvalToolKit


# A patch for qa evaluation
def read_json_qa_gt(gt_file: str):
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for item in data:
        messages = item.get("messages", [])
        videos = item.get("videos", [])
        video_path = videos[0]
        source = item.get("source", [])
        source = source.split("QA_")[-1]
        user_content = None
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content")
        results[user_content + video_path] = source

    return results


# A patch to filter json results based on attribute name in qa eval
def filter_src_data_qa(src_data: dict[str, Any], gt_data: dict[str, Any], attribute: str):
    messages = src_data.get("messages", [])
    videos = src_data.get("videos", [""])
    video_path = videos[0]
    user_content = ""

    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content")

    source = gt_data[user_content + video_path]
    if source == attribute:
        return True
    return False


# A patch for caption evaluation. Directly integrating this function serves no benefits.
def result_evaluate_caption(
    results: list[str],
    output_file: Path,
    methodname: str,
    filename: str,
):
    import re

    vtrs, flus, ras, ias, labs, clus = [], [], [], [], [], []
    attributes = {"vtr": vtrs, "flu": flus, "ra": ras, "ia": ias, "lab": labs, "clu": clus}

    for res in results:
        try:
            vtr, flu, ra, ia, lab, clu = re.findall(r"\[?score[s]?\]?: (\d+)", res.lower())
            vtr, flu, ra, ia, lab, clu = float(vtr), float(flu), float(ra), float(ia), float(lab), float(clu)
            vtrs.append(vtr)
            flus.append(flu)
            ras.append(ra)
            ias.append(ia)
            labs.append(lab)
            clus.append(clu)
        except Exception:
            vtrs.append(0)
            flus.append(0)
            ras.append(0)
            ias.append(0)
            labs.append(0)
            clus.append(0)

    print("vtr", sum(vtrs) / len(vtrs), len(vtrs))
    print("flu", sum(flus) / len(flus), len(flus))
    print("ra", sum(ras) / len(ras), len(ras))
    print("ia", sum(ias) / len(ias), len(ias))
    print("lab", sum(labs) / len(labs), len(labs))
    print("clu", sum(clus) / len(clus), len(clus))

    for attribute, attr_list in attributes.items():
        single_output = output_file.parent / f"{filename}_{attribute}.txt"
        with open(single_output, "a+") as f:
            eval_res = sum(attr_list) / len(attr_list)
            f.write(f"{eval_res},{methodname},{filename}\n")


class EvaluationPipeline:
    def __init__(
        self,
        input_dir: str,
        methodname: str,
        output_dir: str,
        task_metadata: EvalTask,
        *,
        max_concurrency: int = 10,
        retry: int = 20,
    ):
        # Set logger
        self.logger = logging.getLogger(task_metadata.task_name)
        self._set_logging()

        # Set task
        self.task_metadata = task_metadata

        # Set llm
        self.llm = LLMEvalToolKit()

        # Set input file
        filename = self.task_metadata.filename
        self.methodname = methodname
        self.file, _ = os.path.splitext(filename)

        self.input_file = Path(input_dir) / methodname / filename
        self.src_data = self._read_json()

        # Set output file
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / f"{self.file}.txt"

        # Set results
        self.results = []

        self.max_concurrency = max_concurrency
        self.retry = retry

    def _set_logging(self):
        if self.logger.handlers:  # in case of duplicate handlers
            return

        fmt_str = "%(asctime)s.%(msecs)03d %(levelname)7s [%(thread)d][%(process)d] %(message)s"
        fmt = logging.Formatter(fmt_str, datefmt="%H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _read_json(self):
        src_data = []

        with open(self.input_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            src_data.append(json.loads(line))
        self.logger.info(f"Total lines to process: {len(src_data)}")

        return src_data

    def calculate_f1(self, true_label, pred_label, classes: list[str]):
        if self.task_metadata is Tasks.AFF or self.task_metadata is Tasks.CELEBVHQ_APPEARANCE or self.task_metadata is Tasks.CELEBVHQ_ACTION:
            # These tasks require additional handling
            if pred_label.lower() != "none":
                pred_label = [i.lower() for i in pred_label.split(",")]
            else:
                pred_label = []
            true_label = [i.lower() for i in true_label.split(",")]

        y_true_list = [list(true_label)]
        y_pred_list = [list(pred_label)]

        mlb = MultiLabelBinarizer(classes=classes)  # Define classes
        y_true_bin = mlb.fit_transform(y_true_list)
        y_pred_bin = mlb.transform(y_pred_list)

        # f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro")
        f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro")
        # f1_weighted = f1_score(y_true_bin, y_pred_bin, average="weighted")

        return f1_micro * 100

    async def get_response(self, idx: int, data: dict[str, Any]):
        self.logger.info(f"Start processing request {idx}")
        answer = data["response"]
        labels = data["labels"]
        query = data["messages"][0]["content"]
        query = query.replace("<video>.", "")

        process = self.task_metadata.result_process
        if process == "f1":
            prompt = self.task_metadata.prompt.format(query=query, answer=answer)
        else:
            prompt = self.task_metadata.prompt.format(query=query, labels=labels, answer=answer)

        res = await asyncio.to_thread(self.llm, prompt)
        if res is None:
            return res

        res = res.strip()
        self.logger.info(f"Request {idx} done.")

        if process == "f1":
            res = self.calculate_f1(labels, res, self.task_metadata.classes)
            return str(res)
        return res

    async def eval_single(self, idx: int, data: dict[str, Any], sem: asyncio.Semaphore):
        async with sem:
            for i in range(self.retry):
                try:
                    res = await self.get_response(idx, data)
                    if not res:
                        continue
                    elif (
                        '"code": "RateLimitReached"' in res
                        or '"code": "NoCapacity"' in res
                        or '"code": "TooManyRequests"' in res
                        or "Error code: 500" in res
                        or '"code": "ResponsibleAIPolicyViolation"' in res
                        or '"code": "429"' in res
                    ):
                        continue

                    self.results.append(res)
                    return
                except Exception as e:
                    print(f"Attempt {i + 1} {data['videos'][0]} failed: {e}")
                    if i == self.retry - 1:
                        print(data["videos"][0])
                        print("Final attempt failed. No more retries.")
                        return
                await asyncio.sleep(1)

    def result_evaluate(self):
        eval_res = None
        eval_type = self.task_metadata.result_evaluate

        if eval_type == "yn":
            total_num = 0
            for res in self.results:
                if res is None:
                    continue
                if res.lower() == "yes":
                    total_num += 1
            eval_res = total_num / len(self.src_data) * 100
        else:
            func = None
            if eval_type == "int":
                func = int
            elif eval_type == "float":
                func = float

            scores = []
            for res in self.results:
                try:
                    score = func(res)
                except Exception:
                    score = 0
                scores.append(score)

            eval_res = sum(scores) / len(scores)

        with open(self.output_file, "a+") as f:
            f.write(f"{eval_res},{self.methodname},{self.file}\n")

    async def evaluate(self):
        sem = asyncio.Semaphore(self.max_concurrency)

        async with asyncio.TaskGroup() as tg:
            for idx, data in enumerate(self.src_data):
                tg.create_task(self.eval_single(idx, data, sem))

    def execute(self):
        self.logger.info(f"Testing on {self.task_metadata.task_name}")

        if self.task_metadata is Tasks.QA:
            from .config import EvalTaskQA

            self.task_metadata: EvalTaskQA
            gt_results = read_json_qa_gt(self.task_metadata.gt_file)

            original_src = self.src_data

            for attribute in self.task_metadata.attributes:
                self.logger.info(f"Testing on attribute {attribute}")
                _filter_func = partial(filter_src_data_qa, gt_data=gt_results, attribute=attribute)
                self.src_data = filter(_filter_func, original_src)
                asyncio.run(self.evaluate())

                self.output_file = self.output_dir / f"{self.file}_{attribute}.txt"
                self.result_evaluate()
                self.results.clear()

            return

        asyncio.run(self.evaluate())

        if self.task_metadata is Tasks.CAPTION:
            result_evaluate_caption(self.results, self.output_file, self.methodname, self.file)
            return

        self.result_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="method name", required=True, type=str)
    parser.add_argument(
        "--output_dir",
        help="output directory",
        default="eval_face_integrate/results",
        type=str,
    )
    parser.add_argument(
        "--input_dir",
        help="input directory",
        default="result_clean",
        type=str,
    )

    parser.add_argument(
        "--retry",
        help="max retry number",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--max_concurrency",
        help="maximum concurrency",
        default=10,
        type=int,
    )

    args = parser.parse_args()

    for name, task in Tasks.__dict__.items():
        if not isinstance(task, EvalTask):
            continue

        # To evaluate on one specific result, please uncomment the following two lines
        # if task is not Tasks.RAVD_TEST:
        #     continue

        pipeline = EvaluationPipeline(
            input_dir=args.input_dir,
            methodname=args.method,
            output_dir=args.output_dir,
            task_metadata=task,
            max_concurrency=args.max_concurrency,
            retry=args.retry,
        )

        pipeline.execute()
