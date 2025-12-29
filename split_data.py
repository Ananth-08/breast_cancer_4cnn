import os
import shutil
import csv
import re

SOURCE_DIR = "BreaKHis_v1"
DEST_DIR = "data"
CSV_FILE = "patient_split.csv"

MAG_MAP = {
    "-40-": "40X",
    "-100-": "100X",
    "-200-": "200X",
    "-400-": "400X"
}

# =========================
# STEP 1: LOAD CSV
# =========================
patient_map = {}

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    for row in reader:
        if len(row) < 3:
            continue

        full_id = row[0].strip()   # e.g. SOB_M_PC_15-190EF
        cls = row[1].strip().lower()
        split = row[2].strip().lower()

        # extract ONLY core patient id (15-190EF)
        match = re.search(r"\d+-[A-Z0-9]+", full_id)
        if match:
            core_id = match.group()
            patient_map[core_id] = {
                "class": cls,
                "split": split
            }

print(f"Loaded {len(patient_map)} patients from CSV")

# =========================
# STEP 2: COPY FILES
# =========================
copied = 0

for root, _, files in os.walk(SOURCE_DIR):
    for file in files:

        # extract core id from filename
        fid = re.search(r"\d+-[A-Z0-9]+", file)
        if not fid:
            continue

        core_id = fid.group()

        if core_id not in patient_map:
            continue

        # detect magnification
        mag = None
        for k, v in MAG_MAP.items():
            if k in file:
                mag = v
                break

        if mag is None:
            continue

        info = patient_map[core_id]

        src = os.path.join(root, file)
        dst = os.path.join(
            DEST_DIR,
            mag,
            info["split"],
            info["class"]
        )

        os.makedirs(dst, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

print(f"✅ Copied {copied} images successfully!")
