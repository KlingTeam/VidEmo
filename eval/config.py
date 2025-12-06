from typing import List

from dataclasses import dataclass, field


class Settings:
    API_KEY = "YOUR_OPENAI_API_KEY"
    BASE_URL = "YOUR_OPENAI_URL"
    MODEL_NAME = "gpt-4o-2024-08-06"


@dataclass
class EvalTask:
    """
    Base EvalTask Class
    -------------------

    - `task_name`: name of the task
    - `filename`: the output filename of this task (must be an entire filename)
    - `prompt`: prompt to evaluate this task
    - `result_process`: process the GPT result (strip: str.strip, f1: calculating f1 score,
    check `eval_results.EvaluationPipeline.calculate_f1` for details)
    - `result_evaluate`: evaluate the GPT result (int: built-in integer conversion, float:
    built-in float conversion, yn: whether the result is "yes")
    """

    task_name: str
    filename: str
    prompt: str
    result_process: str = field(default="strip", metadata={"help": "strip, f1"})
    result_evaluate: str = field(default="float", metadata={"help": "int, float, yn"})


@dataclass(kw_only=True)
class EvalTaskMultiClass(EvalTask):
    """
    Multi-class Evaluation
    ----------------------
    - `classes`: used for the calculation of f1 score
    """

    classes: List[str]


@dataclass(kw_only=True)
class EvalTaskQA(EvalTask):
    """
    QA Evaluation
    -------------
    - `gt_file`: the location for ground truth annotation
    - `attributes`: all open-ended attributes to evaluate
    """

    gt_file: str
    attributes: List[str]


class PromptTemplate:
    YN = """
    You are an expert in evaluating video facial expression analysis.

    Question: {query}
    The ground true answer: {labels}
    The model's response: {answer}

    Please judge the correctness of model's response according to the given answer and question.

    [Requirement]
    1. Directly output the judgement: yes or no

    [Example]
    yes
    """

    SCORE = """
    You are an expert in evaluating the accuracy of video facial expression caption.

    Question: {query}.
    The Model's Prediction: {answer}
    The ground true label: {labels}

    Please score the model's prediction according to the correctness and matching degree.

    [Requirement]
    1. Directly output the score number from 0 to 100
    2. No sentence or word

    [Example]
    80
    """

    MAFW_MULTI = """
    Based on the question, please convert provided answer into multiple choice options.

    Question: {query}
    The answer: {answer}

    [Requirement]
    1. Directly output the converted options. Only the letters of the options are in output. Don't output like "(B) Surprise".
    2. Don't say like "Here is the output" or 'output:'. 
    3. Each option is directly concatenated together in the output.

    [Output Example]
    ABK
    """

    AFF = """
    Based on the question, please convert provided answer into multiple choice options connected by commas. Output none if the answer is none or invalid. Output phrases from: AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU15, AU23, AU24, AU25, AU26


    Question: {query}
    The answer: {answer}

    [Requirement]
    1. Directly output the converted phrases from answer. Only the letters of the options are in output. Don't output like "(B) Surprise".
    2. Don't say like "Here is the output" or 'output:'. 
    3. Each option is directly concatenated together in the output.

    [Output Example]
    AU1,AU4
    """

    CELEBVHQ_APPEARANCE = """
    Based on the question, please convert the provided answer into multiple phrases connected by commas. output none if the answer is none or invalid. Output phrases from: blurry, male, young, chubby, pale_skin, rosy_cheeks, oval_face, receding_hairline, bald, bangs, black_hair, blonde_hair, gray_hair, brown_hair, straight_hair, wavy_hair, long_hair, arched_eyebrows, bushy_eyebrows, bags_under_eyes, eyeglasses, sunglasses, narrow_eyes, big_nose, pointy_nose, high_cheekbones, big_lips, double_chin, no_beard, 5_o_clock_shadow, goatee, mustache, sideburns, heavy_makeup, wearing_earrings, wearing_hat, wearing_lipstick, wearing_necklace, wearing_necktie, wearing_mask


    Question: {query}
    The answer: {answer}

    [Requirement]
    1. Directly output the converted phrases from answer. Only output the phrases.
    2. Don't say like "Here is the output" or 'output:'. 
    3. Each option is directly concatenated together in the output.

    [Output Example]
    mustache,receding_hairline
    """

    CELEBVHQ_ACTION = """
    Based on the question, please convert the provided answer into multiple phrases connected by commas. output none if the answer is none or invalid. Output phrases from: blow, chew, close_eyes, cough, cry, drink, eat, frown, gaze, glare, head_wagging, kiss, laugh, listen_to_music, look_around, make_a_face, nod, play_instrument, read, shake_head, shout, sigh, sing, sleep, smile, smoke, sneer, sneeze, sniff, talk, turn, weep, whisper, wink, yawn


    Question: {query}
    The answer: {answer}

    [Requirement]
    1. Directly output the converted phrases from answer. Only output the phrases.
    2. Don't say like "Here is the output" or 'output:'. 
    3. Each option is directly concatenated together in the output.

    [Output Example]
    whisper,talk
    """

    CAPTION = """You are an objective and precise evaluator, specializing in rigorously assessing the facial attribute, expression of indivisual(s) in the video, and multimodal understanding abilities of various models.
    ## [Question Start]\n\n{query}\n\n## [Question End]\n\n\n
    ## [Model's Response Start]\n\n{answer}\n\n## [Model's Response End]\n\n\n
    ## [Ground Truth Answer Start]\n\n{labels}\n\n## [Ground Truth Answer End]\n\n\n
    ## [Instruction]\n\n
    The task instruction of the model is to directly conduct fine-grained caption for a video with all the involved subjects. \n\n
    Please evaluate the following aspects of the model's response:\n
    1. Video-Text Relevance: Are the responses closely related to the visual content of the video?\n
    2. Fluency: Are the responses grammatically correct and smoothly articulated?\n
    3. Response Accuracy: Do the responses accurately answer the question\n
    4. Instruction Adherence: Do the responses accurately adhere to the task instruction, directly describe all the subjects in the video by capturing the facial attributes, emotion-related cues, actions and also indicate the environment, lightings, and context, without any additional explanatory prefixes or suffixes?\n
    5. Label Overlap: What is the overlap rate of emotional state descriptions (expressions, emotion status, etc.) between the Model's Response and the Ground Truth Answer based on matching?\n
    6. Clue Overlap: What is the overlap rate of emotion-related clues (facial attributes, action, background, etc.) between the Model's Response and the Ground Truth Answer?\n\n
    For each aspect, provide a brief qualitative evaluation for the model, followed by quantitative score from 1 to 100, where 1 indicates poor performance and 100 indicates excellent performance.\n\n
    The output should be in the following format:\n
    1. Video-Text Relevance: (Qualitative Evaluation), [Score]: (the score of Model for VTR)\n
    2. Fluency: (Qualitative Evaluation), [Score]: (the score of Model for Flu)\n
    3. Response Accuracy: (Qualitative Evaluation), [Score]: (the score of Model for RA)\n
    4. Instruction Adherence: (Qualitative Evaluation), [Score]: (the score of Model for IA)\n
    5. Label Overlap: (Qualitative Evaluation), [Score]: (the score of Model for LO)\n
    6. Clue Overlap: (Qualitative Evaluation), [Score]: (the score of Model for CO)\n
    \n\n
    Please ensure that your evaluations are unbiased.
    Output:"""

    QA = """
    You are an expert in evaluating the accuracy of video facial expression caption.

    Question: {query}.
    The Model's Prediction: {answer}
    The ground true label: {labels}

    Please score the model's prediction according to the correctness and matching degree.

    [Requirement]
    1. Directly output the score number 0 or 100 for choice type question-answer pairs.
    2. Directly output the score number from 0 to 100 for caption type question-answer pairs.
    3. No sentence or word

    [Example]
    80
    """


class Tasks:
    RAVD_TEST = EvalTask(
        task_name="ravd_test",
        filename="RAVDESS_Fine-grained-Emotion-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    PERR_TEST = EvalTask(
        task_name="perr_test",
        filename="PERR_Single-Label-Conversational-Emotion-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MOSI_TEST = EvalTask(
        task_name="mosi_test",
        filename="MOSI_Single-Label-Sentiment-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MOSEI_TEST = EvalTask(
        task_name="mosei_test",
        filename="MOSEI_Single-Label-Sentiment-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MER2023_TEST = EvalTask(
        task_name="mer2023_test",
        filename="MER2023_Single-Label-Emotion-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MELD_TEST = EvalTask(
        task_name="meld_test",
        filename="MELD_Single-Label-Conversational-Emotion-Recognition_new.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MEAD_TEST_EMOTION = EvalTask(
        task_name="meld_test_emotion",
        filename="MEAD_Fine-grained-Emotion-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MEAD_ID = EvalTask(
        task_name="mead_id",
        filename="MEAD_ID.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MEAD_HEAD_POSE = EvalTask(
        task_name="mead_head_pose",
        filename="MEAD_Head-Pose.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MAFW_SINGLE_TEST = EvalTask(
        task_name="mafw_single_test",
        filename="MAFW_Single-Label-Emotion-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    DFEW_SINGLE_TEST = EvalTask(
        task_name="dfew_single_test",
        filename="DFEW_Single-Label-Emotion-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    CHSIMSV1_TEST = EvalTask(
        task_name="chsimsv1_test",
        filename="CHSIMIv1_Fine-grained-Sentiment-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    CELEBVTEXT_EMOTION = EvalTask(
        task_name="celebvtext_emotion",
        filename="CelebV-Text_Emotion-Caption.json",
        prompt=PromptTemplate.SCORE,
        result_evaluate="int",
    )

    CELEBVTEXT_APPEARANCE = EvalTask(
        task_name="celebvtext_appearance",
        filename="CelebV-Text_Appearance-Caption.json",
        prompt=PromptTemplate.SCORE,
        result_evaluate="int",
    )

    CELEBVTEXT_ACTION = EvalTask(
        task_name="celebvtext_action",
        filename="CelebV-Text_Action-Caption.json",
        prompt=PromptTemplate.SCORE,
        result_evaluate="int",
    )

    CAS_TEST = EvalTask(
        task_name="cas_test",
        filename="CAS_Single-Label-Micro-Expression-Recognition.json",
        prompt=PromptTemplate.YN,
        result_evaluate="yn",
    )

    MAFW_MULTI_TEST = EvalTaskMultiClass(
        task_name="mafw_multi_test",
        filename="MAFW_Multiple-Label-Emotion-Recognition.json",
        prompt=PromptTemplate.MAFW_MULTI,
        result_evaluate="float",
        result_process="f1",
        classes=list("ABCDEFGHIJK"),
    )

    AFF = EvalTaskMultiClass(
        task_name="aff_test",
        filename="AffWild2_Action_Unit_Detection.json",
        prompt=PromptTemplate.AFF,
        result_evaluate="float",
        result_process="f1",
        classes=["au1", "au2", "au4", "au6", "au7", "au10", "au12", "au15", "au23", "au24", "au25", "au26"],
    )

    CELEBVHQ_APPEARANCE = EvalTaskMultiClass(
        task_name="celebvhq_appearance",
        filename="CelebV-HQ_Appearance-Recognition.json",
        prompt=PromptTemplate.CELEBVHQ_APPEARANCE,
        result_evaluate="float",
        result_process="f1",
        classes=[
            "blurry",
            "male",
            "young",
            "chubby",
            "pale_skin",
            "rosy_cheeks",
            "oval_face",
            "receding_hairline",
            "bald",
            "bangs",
            "black_hair",
            "blonde_hair",
            "gray_hair",
            "brown_hair",
            "straight_hair",
            "wavy_hair",
            "long_hair",
            "arched_eyebrows",
            "bushy_eyebrows",
            "bags_under_eyes",
            "eyeglasses",
            "sunglasses",
            "narrow_eyes",
            "big_nose",
            "pointy_nose",
            "high_cheekbones",
            "big_lips",
            "double_chin",
            "no_beard",
            "5_o_clock_shadow",
            "goatee",
            "mustache",
            "sideburns",
            "heavy_makeup",
            "wearing_earrings",
            "wearing_hat",
            "wearing_lipstick",
            "wearing_necklace",
            "wearing_necktie",
            "wearing_mask",
        ],
    )

    CELEBVHQ_ACTION = EvalTaskMultiClass(
        task_name="celebvhq_action",
        filename="CelebV-HQ_Action-Recognition.json",
        prompt=PromptTemplate.CELEBVHQ_ACTION,
        result_evaluate="float",
        result_process="f1",
        classes=[
            "blow",
            "chew",
            "close_eyes",
            "cough",
            "cry",
            "drink",
            "eat",
            "frown",
            "gaze",
            "glare",
            "head_wagging",
            "kiss",
            "laugh",
            "listen_to_music",
            "look_around",
            "make_a_face",
            "nod",
            "play_instrument",
            "read",
            "shake_head",
            "shout",
            "sigh",
            "sing",
            "sleep",
            "smile",
            "smoke",
            "sneer",
            "sneeze",
            "sniff",
            "talk",
            "turn",
            "weep",
            "whisper",
            "wink",
            "yawn",
        ],
    )

    # Caption evaluation has a result-evaluate patch. Thus, we don't set the result_evaluate here.
    CAPTION = EvalTask(
        task_name="emotion_understanding_caption",
        filename="caption_test_sampling_v2_all_example.json",
        prompt=PromptTemplate.CAPTION,
    )

    QA = EvalTaskQA(
        task_name="open_attribute_qa",
        filename="QA_test_sampling_all.json",
        prompt=PromptTemplate.QA,
        gt_file="anno/QA_test_sampling_all/QA_test_sampling_all.json",
        result_evaluate="int",
        attributes=[
            "Gender",
            "Body_actions",
            "Skin",
            "Facial_features",
            "Nose",
            "Eyes",
            "Age",
            "Hair",
            "Facial_actions",
            "Accessories",
            "Eyebrows",
            "Head_actions",
            "Chin",
            "Mouth",
            "Face_shape",
        ],
    )
