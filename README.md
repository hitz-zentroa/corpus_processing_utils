# corpus_processing_utils

Tools for preparing, cleaning, analyzing, and evaluating speech corpora.

This repository provides utility scripts for processing audio--text
datasets, computing corpus statistics, normalizing text, and evaluating
ASR systems using Word Error Rate (WER).\
The utilities are designed to be modular, language-aware (Spanish and
Basque), and compatible with JSONL manifests commonly used in speech
processing pipelines.

## Features

### **1. Manifest & Corpus Utilities (`corpus_utils.py`)**

-   Read/write manifests in **JSON Lines** format\
-   Convert TSV datasets to structured manifest dictionaries\
-   Pair `.txt` transcript files with `.wav` audio files\
-   Compute hashes & deduplicate corpora\
-   Reduce corpora using reference datasets\
-   Compute duration statistics\
-   Export statistics or WER results to **Excel (.xlsx)**

### **2. Text Normalization (`normalizer.py`)**

-   Language-specific normalization (Spanish `es`, Basque `eu`)\
-   Optional case/punctuation preservation\
-   Removal of diacritics, unwanted characters, acronyms\
-   Duration-based filtering\
-   Blacklist-based filtering\
-   Detailed logging of removed entries and character distributions

### **3. WER Evaluation (`wer_evaluator.py`)**

-   Compute **sentence-level** and **corpus-level** WER\
-   Optional evaluation with **case-preserving** (C&P) text\
-   Uses the same normalization pipeline as training\
-   Outputs both cleaned manifests and WER summaries\
-   Compatible with JSONL ASR output manifests

## Installation
Recommended dependencies:

    pandas
    openpyxl
    soundfile
    tqdm
    jiwer

## Repository Structure

    corpus_processing_utils/
    │
    ├── corpus_utils.py
    ├── normalizer.py
    ├── wer_evaluator.py
    └── README.md

## Usage Examples

### 1. Reading / Writing a Manifest

``` python
import corpus_utils as cu

data = cu.read_manifest("data/train.json")
cu.write_manifest("out/train_clean.json", data)
```

### 2. Convert a TSV file to a manifest structure

``` python
from corpus_utils import tsv2data

data = tsv2data(
    "dataset.tsv",
    clips_folder="clips/",
    audio_field="path",
    text_field="sentence",
    calculate_duration=True
)
```

### 3. Text Normalization

``` python
from normalizer import TextNormalizer

normalizer = TextNormalizer(
    lang="es",
    keep_cp=False,
    remove_acronyms=True
)

clean_data = normalizer(data)
```

### 4. Compute WER for a manifest

``` python
from wer_evaluator import calculate_wer

clean_data, wer_stats = calculate_wer(
    "predictions.json",
    lang="es",
    cp_field=True,
    return_wer=True
)
```

### 5. Export statistics to Excel

``` python
from corpus_utils import manifest_time_stats, stats2xlsx

stats = manifest_time_stats("data.json", return_stats=True)
stats2xlsx([stats], "stats.xlsx")
```

## Main Functionalities

### **corpus_utils.py**

-   Manifest reading/writing\
-   TSV conversion\
-   File pairing\
-   Hashing & deduplication\
-   Duration statistics\
-   Excel exporting

### **normalizer.py**

-   Language rules\
-   Case normalization\
-   Punctuation cleaning\
-   Duration filtering\
-   Acronym removal\
-   Verbose mode

### **wer_evaluator.py**

-   WER calculation (mean & total)\
-   Optional C&P analysis\
-   Manifest cleaning\
-   Summary export

## Contributions

PRs and issues are welcome!
