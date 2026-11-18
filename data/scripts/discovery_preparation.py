"""Deterministic historical discovery preprocessing; no model calls or generation."""

import re
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from remove_reset.artifacts import read_json as load_json
from remove_reset.artifacts import value_hash
from remove_reset.config import MODEL_ID, REVISION

protocol = SimpleNamespace(value_hash=value_hash, MODEL_ID=MODEL_ID, MODEL_REVISION=REVISION)


FORMAT = "cross-domain-originals-v1"

SCREENING_PROTOCOL = "original-base-support-v3"

SEED = 20261024

NATURAL_REVISION = "55a365637381ce7f3748fa2eac7aef1a113bbb82"

BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"

LICENSES = {
    "MIT": "MIT",
    "Apache 2.0": "Apache-2.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
}

NATURAL_TASKS = {
    "task002_quoref_answer_generation": ("coreference_reading", "encyclopedic_reading"),
    "task075_squad1.1_answer_generation": ("extractive_qa", "encyclopedic_reading"),
    "task190_snli_classification": ("entailment", "everyday_language"),
    "task349_squad2.0_answerable_unanswerable_question_classification": (
        "answerability",
        "encyclopedic_reading",
    ),
    "task380_boolq_yes_no_question": ("passage_yes_no", "encyclopedic_reading"),
    "task1391_winogrande_easy_answer_generation": ("coreference_choice", "everyday_language"),
    "task1548_wiqa_binary_classification": ("process_ordering", "physical_science"),
    "task1727_wiqa_what_is_the_effect": ("causal_effect", "physical_science"),
}

NATURAL_LABELS = {
    "task190_snli_classification": ["E", "C", "N"],
    "task349_squad2.0_answerable_unanswerable_question_classification": ["True", "False"],
    "task380_boolq_yes_no_question": ["Yes", "No"],
    "task1391_winogrande_easy_answer_generation": ["A", "B"],
    "task1548_wiqa_binary_classification": ["Yes", "No"],
    "task1727_wiqa_what_is_the_effect": ["for", "against", "none"],
}

LICENSE_REVIEW_TASKS = [
    "task363_sst2_polarity_classification",
    "task1289_trec_classification",
    "task199_mnli_classification",
    "task591_sciq_answer_generation",
    "task625_xlwic_true_or_false_answer_generation",
]

BBH_TASKS = {
    "boolean_expressions": ("boolean_logic", "formal_reasoning"),
    "date_understanding": ("date_reasoning", "everyday_planning"),
    "disambiguation_qa": ("coreference_choice", "everyday_language"),
    "formal_fallacies": ("deduction", "formal_reasoning"),
    "logical_deduction_three_objects": ("ordering", "formal_reasoning"),
    "object_counting": ("counting", "mathematics"),
    "temporal_sequences": ("temporal_reasoning", "everyday_planning"),
    "sports_understanding": ("plausibility", "sports"),
    "word_sorting": ("word_sorting", "language_form"),
    "tracking_shuffled_objects_three_objects": ("state_tracking", "formal_reasoning"),
}

MMLU_SUBJECTS = {
    "anatomy": "biology_health",
    "astronomy": "physical_science",
    "business_ethics": "business",
    "clinical_knowledge": "biology_health",
    "computer_security": "computing",
    "conceptual_physics": "physical_science",
    "elementary_mathematics": "mathematics",
    "high_school_biology": "biology_health",
    "high_school_chemistry": "physical_science",
    "high_school_computer_science": "computing",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "civic_knowledge",
    "high_school_macroeconomics": "economics",
    "high_school_microeconomics": "economics",
    "high_school_psychology": "psychology",
    "high_school_world_history": "history",
    "human_aging": "biology_health",
    "management": "business",
    "philosophy": "humanities",
    "world_religions": "humanities",
}

HF_SOURCES = {
    "mmlu": ("cais/mmlu", "mit", ["all/validation-00000-of-00001.parquet"]),
    "arc": (
        "allenai/ai2_arc",
        "cc-by-sa-4.0",
        [
            "ARC-Easy/train-00000-of-00001.parquet",
            "ARC-Challenge/train-00000-of-00001.parquet",
        ],
    ),
    "gsm8k": ("openai/gsm8k", "mit", ["main/train-00000-of-00001.parquet"]),
    "mbpp": (
        "google-research-datasets/mbpp",
        "cc-by-4.0",
        [
            "sanitized/train-00000-of-00001.parquet",
        ],
    ),
}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def canonical_answer(text):
    return normalized(text).strip(" \t\n\r\"'`.,;:!?")


def labels_in(instances, field):
    labels = sorted({str(answer).strip() for row in instances for answer in field(row)})
    return labels if 1 < len(labels) <= 8 and max(map(len, labels)) <= 40 else []


def group_identity(source, task, text):
    patterns = [
        r"(?is)^passage:\s*(.*?)\s*(?:\n\s*question:| question:)",
        r"(?is)^process:\s*(.*?)\s*question:",
        r"(?is)^sentence 1:\s*(.*?)\s*sentence 2:",
    ]
    namespace = source
    if source == "natural":
        if "squad" in task:
            namespace += ":squad"
        elif "wiqa" in task:
            namespace += ":wiqa"
        else:
            namespace += ":" + task.split("_", 2)[1]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return protocol.value_hash([namespace, normalized(match[1])])
    return protocol.value_hash([namespace, normalized(text)])


def partition(group_id):
    value = int(protocol.value_hash([FORMAT, "partition", group_id])[:16], 16) / 2**64
    return "pilot" if value < 0.2 else "score" if value < 0.8 else "audit"


def family(
    source,
    task,
    row_id,
    original,
    answers,
    *,
    domain,
    kind,
    quota,
    options=None,
    instruction="",
    details=None,
    code_tests=None,
):
    if not isinstance(original, str) or not original.strip():
        raise ValueError(f"empty/nontext source input: {source}/{task}/{row_id}")
    if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError(f"invalid source answers: {source}/{task}/{row_id}")
    group_id = group_identity(source, task, original)
    family_id = f"{source}:{task}:{row_id}"
    row = {
        "format": FORMAT,
        "family_id": family_id,
        "variant_id": family_id + ":original:en",
        "source": source,
        "source_task": task,
        "source_row_id": str(row_id),
        "domain": domain,
        "task_type": kind,
        "original_input": original,
        "instruction": instruction,
        "answers": [answer.strip() for answer in answers],
        "options": options or [],
        "group_id": group_id,
        "partition": partition(group_id),
        "language": "en",
        "transformation": "original",
        "quota_per_task": quota,
        "translation_eligibility": "ineligible" if task == "word_sorting" else "requires_review",
        "details": details or {},
    }
    if code_tests is not None:
        row["code_tests"] = code_tests
    row["input"] = render(row)
    row["input_sha256"] = protocol.value_hash(row["input"])
    row["content_sha256"] = protocol.value_hash(
        [normalized(original), sorted(canonical_answer(a) for a in answers)]
    )
    return row


def render(row):
    text = row["original_input"]
    if row["source"] in ("mmlu", "arc"):
        choices = "\n".join(f"{option['label']}. {option['text']}" for option in row["options"])
        return f"Question: {text}\n{choices}\nAnswer:"
    if row["source"] == "natural":
        return f"Task: {row['instruction']}\nInput: {text}\nAnswer:"
    if row["source"] == "mbpp":
        tests = "\n".join(row["code_tests"])
        return f"Task: {text}\nTests:\n{tests}\nPython code:\n"
    if row["source"] == "gsm8k":
        demonstrations = "\n\n".join(
            f"Question: {demo['question']}\nAnswer:\n{demo['answer']}"
            for demo in row["details"]["demonstrations"]
        )
        return demonstrations + f"\n\nQuestion: {text}\nAnswer:\n"
    return f"Question: {text}\nAnswer:"


def source_rows(directory, source):
    import pyarrow.parquet as parquet

    name = source["source"]
    rows = []
    for entry in source["files"]:
        path = directory / entry["path"]
        if name in HF_SOURCES:
            records = parquet.read_table(path).to_pylist()
            for index, row in enumerate(records):
                details = {"file": entry["path"], "original_split": "train", "raw_row_index": index}
                if name == "mmlu":
                    subject = row["subject"]
                    if subject not in MMLU_SUBJECTS:
                        continue
                    details["original_split"] = "validation"
                    options = [
                        {"label": chr(65 + i), "text": value}
                        for i, value in enumerate(row["choices"])
                    ]
                    item = family(
                        name,
                        subject,
                        index,
                        row["question"],
                        [chr(65 + row["answer"])],
                        domain=MMLU_SUBJECTS[subject],
                        kind="subject_knowledge",
                        quota=5,
                        options=options,
                        details=details,
                    )
                elif name == "arc":
                    task = Path(entry["path"]).parts[1]
                    choices = row["choices"]
                    options = [
                        {"label": label, "text": text}
                        for label, text in zip(choices["label"], choices["text"], strict=True)
                    ]
                    if row["answerKey"] not in [option["label"] for option in options]:
                        raise ValueError("ARC source answer is missing from choices")
                    item = family(
                        name,
                        task,
                        row["id"],
                        row["question"],
                        [row["answerKey"]],
                        domain="school_science",
                        kind="science_qa",
                        quota=30,
                        options=options,
                        details=details,
                    )
                elif name == "gsm8k":
                    if index < 2:
                        continue
                    answer = row["answer"].rsplit("####", 1)
                    if len(answer) != 2:
                        raise ValueError("GSM8K source lacks final answer marker")
                    details["reference_solution"] = row["answer"]
                    details["demonstrations"] = [
                        {
                            "raw_row_index": i,
                            "question": records[i]["question"],
                            "answer": records[i]["answer"],
                        }
                        for i in range(2)
                    ]
                    item = family(
                        name,
                        "main",
                        index,
                        row["question"],
                        [answer[1].strip()],
                        domain="mathematics",
                        kind="arithmetic_word_problem",
                        quota=60,
                        details=details,
                    )
                else:
                    details["test_imports"] = row["test_imports"]
                    item = family(
                        name,
                        "sanitized",
                        row["task_id"],
                        row["prompt"],
                        [row["code"]],
                        domain="computing",
                        kind="code_generation",
                        quota=40,
                        details=details,
                        code_tests=row["test_list"],
                    )
                rows.append(item)
        elif name == "natural":
            data = load_json(path)
            task = source["task"]
            labels = NATURAL_LABELS.get(task, [])
            kind, domain = NATURAL_TASKS[task]
            for row in data["Instances"]:
                rows.append(
                    family(
                        name,
                        task,
                        row["id"],
                        row["input"],
                        row["output"],
                        domain=domain,
                        kind=kind,
                        quota=15,
                        instruction="\n".join(data["Definition"]),
                        options=[{"label": label, "text": label} for label in labels],
                        details={
                            "file": entry["path"],
                            "original_split": "Instances",
                            "original_sources": data["Source"],
                            "original_urls": data["URL"],
                            "source_domains": data["Domains"],
                            "source_categories": data["Categories"],
                        },
                    )
                )
        elif name == "bbh":
            task = path.stem
            records = load_json(path)["examples"]
            labels = labels_in(records, lambda row: [row["target"]])
            kind, domain = BBH_TASKS[task]
            for index, row in enumerate(records):
                rows.append(
                    family(
                        name,
                        task,
                        index,
                        row["input"],
                        [row["target"]],
                        domain=domain,
                        kind=kind,
                        quota=8,
                        options=[{"label": label, "text": label} for label in labels],
                        details={"file": entry["path"], "original_split": "published_bbh"},
                    )
                )
        else:
            raise ValueError(f"unsupported acquired source: {name}")
    for row in rows:
        row["provenance"] = {
            "revision": source["revision"],
            "license": source["license"],
            "license_evidence": source["license_evidence"]["path"],
            "source_file": row["details"]["file"],
            "source_file_sha256": next(
                entry["sha256"]
                for entry in source["files"]
                if entry["path"] == row["details"]["file"]
            ),
            "source_url": next(
                entry["url"] for entry in source["files"] if entry["path"] == row["details"]["file"]
            ),
        }
    return rows


def coverage(rows):
    return {
        key: dict(sorted(Counter(str(row[key]) for row in rows).items()))
        for key in ("source", "source_task", "task_type", "domain", "partition")
    }


def validate_families(rows):
    ids, contents, groups = set(), set(), {}
    for row in rows:
        if row["family_id"] in ids or row["content_sha256"] in contents:
            raise ValueError("duplicate family/content")
        ids.add(row["family_id"])
        contents.add(row["content_sha256"])
        if groups.setdefault(row["group_id"], row["partition"]) != row["partition"]:
            raise ValueError("a group crosses partitions")
        if row["partition"] != partition(row["group_id"]):
            raise ValueError("incorrect deterministic partition")
        if row["transformation"] != "original" or row["language"] != "en":
            raise ValueError("this corpus contains originals only")
        if row["input"] != render(row) or protocol.value_hash(row["input"]) != row["input_sha256"]:
            raise ValueError("rendered input/provenance mismatch")
        labels = [item["label"] for item in row["options"]]
        if labels and (len(labels) != len(set(labels)) or row["answers"][0] not in labels):
            raise ValueError("invalid answer-label vocabulary")
    return {"families": len(ids), "groups": len(groups), "group_overlap": 0}


def select_supported(candidates, outcomes):
    by_id = {row["family_id"]: row for row in outcomes}
    if len(by_id) != len(outcomes) or set(by_id) != {row["family_id"] for row in candidates}:
        raise ValueError("baseline outcomes do not exactly cover candidates")
    counts, accepted, decisions = Counter(), [], []
    for row in candidates:
        outcome = by_id[row["family_id"]]
        key = row["source"], row["source_task"]
        if not outcome["correct"]:
            reason = (
                outcome["status"]
                if outcome["status"] not in ("scored", "generated")
                else "incorrect"
            )
        elif counts[key] >= row["quota_per_task"]:
            reason = "correct_surplus"
        else:
            reason = "accepted"
            counts[key] += 1
            accepted.append(
                {
                    **row,
                    "baseline": {
                        "model_id": protocol.MODEL_ID,
                        "revision": protocol.MODEL_REVISION,
                        "correct": True,
                        "method": outcome["method"],
                        "outcome_sha256": protocol.value_hash(outcome),
                        "support_scope": "greedy_label_and_likelihood_agreement"
                        if row["options"]
                        else "unrestricted_completion_under_recorded_format",
                    },
                }
            )
        decisions.append({"family_id": row["family_id"], "decision": reason})
    return accepted, decisions
