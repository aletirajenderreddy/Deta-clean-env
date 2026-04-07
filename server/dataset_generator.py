from __future__ import annotations

import copy
import random
from datetime import date
from typing import Dict, List

from faker import Faker


COLUMNS = ["id", "name", "age", "email", "phone", "salary", "hire_date", "department", "is_active"]
DEPARTMENTS = ["Engineering", "Finance", "Sales", "HR", "Operations", "Marketing"]
TASK_SIZES = {"easy": 8, "medium": 10, "hard": 12}
TASK_SEEDS = {"easy": 101, "medium": 202, "hard": 303}


def _canonical_email(name: str) -> str:
    parts = [part.lower() for part in name.replace("'", "").split()]
    if len(parts) == 1:
        return f"{parts[0]}@example.com"
    return f"{parts[0]}.{parts[-1]}@example.com"


def _canonical_phone(rng: random.Random) -> str:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(10))
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def _clean_row(index: int, faker: Faker, rng: random.Random) -> dict:
    name = faker.name()
    hire_date = faker.date_between(start_date="-10y", end_date="-30d")
    return {
        "id": 1000 + index,
        "name": name.title(),
        "age": rng.randint(24, 58),
        "email": _canonical_email(name),
        "phone": _canonical_phone(rng),
        "salary": rng.randint(55000, 180000),
        "hire_date": hire_date.strftime("%Y-%m-%d"),
        "department": rng.choice(DEPARTMENTS),
        "is_active": rng.choice([True, False]),
    }


def _build_ground_truth(task_level: str) -> List[dict]:
    seed = TASK_SEEDS[task_level]
    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)
    return [_clean_row(index, faker, rng) for index in range(1, TASK_SIZES[task_level] + 1)]


def _inject_easy_issues(rows: List[dict]) -> Dict[str, int]:
    rows[1]["age"] = None
    rows[3]["email"] = ""
    rows[5]["age"] = ""
    rows.append(copy.deepcopy(rows[2]))
    rows[0]["hire_date"] = rows[0]["hire_date"].replace("-", "/")
    rows[4]["hire_date"] = rows[4]["hire_date"].replace("-", "/")
    return {"missing_values": 3, "duplicates": 1, "wrong_format": 2}


def _inject_medium_issues(rows: List[dict]) -> Dict[str, int]:
    rows[0]["age"] = None
    rows[2]["email"] = ""
    rows[6]["salary"] = ""

    rows.append(copy.deepcopy(rows[1]))
    rows.append(copy.deepcopy(rows[5]))

    rows[0]["hire_date"] = rows[0]["hire_date"].replace("-", "/")
    rows[3]["hire_date"] = f"{rows[3]['hire_date'][8:10]}-{rows[3]['hire_date'][5:7]}-{rows[3]['hire_date'][0:4]}"
    # Format phone rows that still have valid string values
    rows[4]["phone"] = rows[4]["phone"].replace("-", "")   # strip dashes
    rows[7]["phone"] = rows[7]["phone"].replace("-", " ")  # replace dashes with spaces
    # Now set one phone field to None (missing value)
    rows[9]["phone"] = None

    rows[1]["age"] = "thirty"
    rows[8]["age"] = "30.0"

    rows[2]["email"] = "invalid.email.example.com"
    return {
        "missing_values": 4,   # rows 0(age), 2(email), 6(salary), 9(phone)
        "duplicates": 2,       # rows appended copies of 1 and 5
        "wrong_format": 4,     # rows 0(hire_date), 3(hire_date), 4(phone), 7(phone)
        "wrong_type": 2,       # rows 1(age="thirty"), 8(age="30.0")
        "invalid_email": 1,    # row 2 (already cleared, overwritten with bad email)
    }


def _inject_hard_issues(rows: List[dict]) -> Dict[str, int]:
    issues = _inject_medium_issues(rows)

    rows[0]["age"] = 999
    rows[3]["salary"] = -50000
    rows[6]["age"] = -5
    rows[8]["salary"] = 0.01

    near_duplicate_a = copy.deepcopy(rows[2])
    near_duplicate_a["name"] = near_duplicate_a["name"].replace(" ", "  ")
    near_duplicate_a["department"] = "engineering"

    near_duplicate_b = copy.deepcopy(rows[4])
    near_duplicate_b["name"] = near_duplicate_b["name"].lower()
    near_duplicate_b["is_active"] = "yes"

    near_duplicate_c = copy.deepcopy(rows[7])
    near_duplicate_c["name"] = near_duplicate_c["name"].upper()
    near_duplicate_c["department"] = "Eng"

    rows.extend([near_duplicate_a, near_duplicate_b, near_duplicate_c])

    rows[1]["department"] = "Eng"
    rows[5]["department"] = "engineering"
    rows[9]["department"] = "hr"

    rows[10]["is_active"] = "1"
    rows[11]["is_active"] = "no"

    rows[2]["name"] = rows[2]["name"].lower()
    rows[7]["name"] = rows[7]["name"].upper()

    issues.update(
        {
            "outliers": 4,
            "near_duplicates": 3,
            "inconsistent_values": 5,
        }
    )
    return issues


def generate_dataset(task_level: str) -> Dict:
    if task_level not in TASK_SIZES:
        raise ValueError(f"Unsupported task_level '{task_level}'. Use easy, medium, or hard.")

    ground_truth = _build_ground_truth(task_level)
    messy = copy.deepcopy(ground_truth)

    if task_level == "easy":
        issues = _inject_easy_issues(messy)
    elif task_level == "medium":
        issues = _inject_medium_issues(messy)
    else:
        issues = _inject_hard_issues(messy)

    return {"messy": messy, "ground_truth": ground_truth, "issues": issues}
