import json
import os
import statistics
import openpyxl
import logging
import pandas as pd
from tqdm import tqdm
import soundfile as sf

def read_manifest(manifest_filepath, verbose: bool = True):
    """
    Read a manifest file (JSONL format) into a list of dictionaries.

    Parameters
    ----------
    manifest_filepath : str
        Path to the manifest file. Each line of the file should be a valid JSON object.
    
    verbose : bool, optional (default=True)
        If True, logs a message indicating which file is being read.

    Returns
    -------
    list of dict
        A list where each element is a dictionary representing one line from the manifest.

    Raises
    ------
    Exception
        If the file cannot be opened or read.
    
    Notes
    -----
    The manifest file is expected to be in JSON Lines format (one JSON object per line).
    """
    if verbose==True:
        logging.info(f"Reading: {manifest_filepath}")
    try:
        f = open(manifest_filepath, 'r', encoding='utf-8')
        data = [json.loads(line) for line in f]
        f.close()
        return data
    except:
        raise Exception(f"Manifest file could not be opened: {manifest_filepath}")

def write_manifest(manifest_filepath, data, ensure_ascii: bool = False, return_manifest_filepath: bool = False, verbose: bool = True):
    """
    Write a list of dictionaries to a manifest file in JSONL format.

    Parameters
    ----------
    manifest_filepath : str
        Path where the manifest file will be written.
    
    data : list of dict
        List of dictionaries to write to the file. Each dictionary will be written as a JSON object on a separate line.
    
    ensure_ascii : bool, optional (default=False)
        If True, the output will have all non-ASCII characters escaped. Otherwise, non-ASCII characters are written as-is.

    return_manifest_filepath : bool, optional (default=False)
        If True, the function returns the path of the written manifest file. Otherwise, returns None.

    verbose : bool, optional (default=True)
        If True, logs a message when writing is finished.

    Returns
    -------
    str or None
        The manifest file path if `return_manifest_filepath=True`, else None.

    Notes
    -----
    This function overwrites the file if it already exists.
    Each dictionary in `data` is written as a single line in JSON format.
    """
    f = open(manifest_filepath, "w", encoding="utf-8")
    for item in data:
        f.write(json.dumps(item,ensure_ascii=ensure_ascii) + "\n")
    f.close()
    if verbose==True:
        logging.info(f"End Writing manifest: {manifest_filepath}")
    if return_manifest_filepath:
        return manifest_filepath
    else:
        return None

def tsv2data(tsv_filepath: str, clips_folder: str="", sep: str="\t", audio_field="path", text_field="sentence", duration_field=None, calculate_duration: bool=False, header='infer'):
    """
    Reads a TSV file containing audio file paths and corresponding text sentences,
    returning a structured list of dictionaries suitable for downstream processing.

    Each row in the TSV represents a single audio-text pair. Optionally, if a column
    for precomputed durations is provided, it will be used; otherwise, durations can
    be calculated directly from the audio files using the `soundfile` library.

    Parameters
    ----------
    tsv_filepath : str
        Path to the TSV file to read.

    clips_folder : str, optional (default="")
        Base folder to prepend to audio file paths. Useful if the paths in the TSV
        are relative.

    sep : str, optional (default="\t")
        Column separator in the TSV file.

    audio_field : str or int, optional (default="path")
        Column containing audio file paths. Can be a string (column name) if headers
        exist, or an integer index if the TSV has no headers.

    text_field : str or int, optional (default="sentence")
        Column containing the corresponding text for each audio file. Can be a string
        or integer index.

    duration_field : str or int, optional (default=None)
        Column containing precomputed duration values. If not provided, durations
        will be calculated if `calcuate_duration` is True.

    calculate_duration : bool, optional (default=False)
        Whether to compute audio durations from the audio files using `soundfile`.
        If False, duration will be set to None unless `duration_field` is provided.

    header : int, list of int, 'infer', or None, optional (default='infer')
        Row(s) to use as the column names. Follows the same convention as `pandas.read_csv`.
        Set to None if the TSV has no headers.

    Returns
    -------
    list of dict
        Each dictionary contains:
            - 'audio_filepath': full path to the audio file (str)
            - 'text': corresponding text (str)
            - 'duration': duration in seconds (float) or None if not calculated

    Notes
    -----
    - Raises an Exception if an audio file cannot be opened when duration calculation
      is enabled.
    - The `clips_folder` is prepended to each audio path using `os.path.join`.
    """
    data=[]
    df = pd.read_csv(tsv_filepath, sep=sep, header=header)
    for idx in tqdm(range(len(df))):
        audio_filepath = os.path.join(clips_folder,df[audio_field][idx])
        text = df[text_field][idx]
        if calculate_duration:
            if duration_field:
                try:
                    duration = df[duration_field][idx]
                except:
                    raise Exception(f"Audio file could not be opened: {audio_filepath}")
            else:
                f = sf.SoundFile(audio_filepath)
                duration = len(f) / f.samplerate
        else:
            duration = None
        item = {
            'audio_filepath': audio_filepath,
            'text': text,
            'duration': duration
        }
        data.append(item)
    return data
    
def pairedfiles2data(clips_folder, sentences_folder):
    """
    Build a structured dataset by pairing text files with their corresponding audio files.

    This function scans `sentences_folder` for `.txt` files and expects each one to have
    a `.wav` file with the same base name in `clips_folder`. For every valid pair, it
    loads the text, opens the audio file to retrieve its duration, and returns a dataset
    where each entry contains:

        - audio_filepath : full path to the `.wav` file
        - text           : sentence string loaded from the `.txt` file
        - duration       : audio duration in seconds

    Parameters
    ----------
    clips_folder : str
        Path to the directory containing `.wav` audio files.
    sentences_folder : str
        Path to the directory containing `.txt` sentence files.

    Returns
    -------
    list of dict
        A list of paired items. Each dictionary contains:
            {
                'audio_filepath': str,
                'text': str,
                'duration': float
            }

    Raises
    ------
    Exception
        If a corresponding audio file cannot be opened or does not exist.

    Notes
    -----
    - `.txt` filenames must match the `.wav` filenames (same stem).
    - Duration is obtained using `sf.SoundFile`, which must be available and functional.
    """
    data=[]
    for file in os.listdir(sentences_folder):
        if file.endswith(".txt"):
            audio_filepath = os.path.join(clips_folder,file.replace(".txt",".wav"))
            try:
                duration = sf.SounFile(audio_filepath)
            except:
                raise Exception(f"Audio file could not be opened: {audio_filepath}")
            f = open(os.path.join(sentences_folder,file),"r",encoding="utf-8")
            sentence = f.read()
            item = {
                'audio_filepath': audio_filepath,
                'text': sentence,
                'duration': duration,
            }
            data.append(item)
    return data

def hash_sentences(data):
    """
    Compute hash values for the `"text"` field of each item in a dataset.

    Parameters
    ----------
    data : list of dict
        A list of items where each item must contain a `"text"` key.

    Returns
    -------
    list of int
        A list of hash values corresponding to `hash(item["text"])` for each item
        in `data`.

    Notes
    -----
    - The built-in `hash()` function is used, so hash values may differ across
      Python sessions unless hash randomization is disabled.
    - This function is typically used to accelerate duplicate detection or
      cross-dataset comparisons.
    """
    hashed_sentences = [hash(item["text"]) for item in tqdm(data)]
    return hashed_sentences

def reduce_data(data, compare_data=None, hashed_data=None, hashed_compare=None):
    """
    Reduce a dataset by removing duplicates or by filtering out items
    that appear in another dataset. Hashes of the `"text"` field are used
    for fast comparison.

    Parameters
    ----------
    data : list of dict
        The primary dataset to reduce. Each item must contain a `"text"` key.
    compare_data : list of dict, optional
        If provided, items from `data` whose `"text"` hashes appear in
        `compare_data` will be removed. If None, the function removes
        duplicates within `data` itself.
    hashed_data : list of int, optional
        Precomputed hash values for each item in `data`. Must be aligned by index.
        If not provided, hashes are computed internally as `hash(item["text"])`.
    hashed_compare : list of int, optional
        Precomputed hash values for each item in `compare_data`. Used only when
        `compare_data` is provided. If None, hashes are computed automatically.

    Returns
    -------
    list of dict
        The reduced dataset with duplicates removed or filtered based on
        `compare_data`.

    Notes
    -----
    - Duplicate detection relies solely on the hash of the `"text"` field.
      If two items have identical text, only the first is kept.
    - When `compare_data` is provided, the function removes all `data` items
      whose `"text"` hash is found in `compare_data`.
    - The function logs the number and percentage of removed items.
    """
    logging.info("::::: Reducing dataset :::::")
    if hashed_data is None:
        hashed_data = [hash(item["text"]) for item in tqdm(data, desc="Hashing data")]
    datalen = len(data)
    if compare_data is None:
        # Remove duplicates within the same dataset
        seen_hashes = set()
        reduced_data = []
        for h, item in tqdm(zip(hashed_data, data), total=datalen, desc="Removing duplicates"):
            if h not in seen_hashes:
                reduced_data.append(item)
                seen_hashes.add(h)
    else:
        # Remove items in data that exist in compare_data
        if hashed_compare is None:
            hashed_compare = [hash(item["text"]) for item in tqdm(compare_data, desc="Hashing compare_data")]
        hashed_compare_set = set(hashed_compare)
        reduced_data = [item for h, item in tqdm(zip(hashed_data, data), total=datalen, desc="Filtering compare_data") if h not in hashed_compare_set]
    removed_count = datalen - len(reduced_data)
    logging.info(f"- Removed: {removed_count}/{datalen} ({round(100*removed_count/datalen, 2)}%)")
    return reduced_data

def manifest_time_stats(manifest, return_stats: bool = False, verbose: bool = True):
    """
    Compute duration statistics from a manifest and optionally print them.

    Parameters
    ----------
    manifest : str or list
        - If a string: treated as a filepath to a manifest JSON/JSONL file
          readable by `read_manifest()`.  
        - If a list: assumed to be an already-loaded list of dicts where each
          item contains a `"duration"` field.

    return_stats : bool, optional (default=False)
        If True, return a dictionary containing all computed statistics.

    verbose : bool, optional (default=True)
        If True, log the statistics to the console using `logging.info()`.

    Returns
    -------
    dict (optional)
        Returned only if `return_stats=True`. Contains:
            - "filename": source of the manifest  
            - "t_min": minimum segment duration (s)  
            - "t_mean": mean segment duration (s)  
            - "t_median": median segment duration (s)  
            - "t_max": maximum segment duration (s)  
            - "t_total": [sum of durations in seconds, hours]  
            - "t_total_median": [median * count in seconds, hours]  
            - "sentences": number of entries in the manifest

    Notes
    -----
    The function expects each entry in the manifest to contain a
    `"duration"` key whose value can be converted to a float.
    """
    if isinstance(manifest, str):
        data = read_manifest(manifest)
        filename = os.path.split(manifest)[1]
    elif isinstance(manifest, list):
        data = manifest
        filename = "in-memory data"
    else:
        raise Exception(f"ERROR: 'manifest' must be 'str' or 'list'")
    duration = [float(item['duration']) for item in data]
    stats = {
        "filename": filename,
        "t_min": round(min(duration),2),
        "t_mean": round(statistics.mean(duration),2),
        "t_median": round(statistics.median(duration),2),
        "t_max": round(max(duration),2),
        "t_total": [round(sum(duration),2), round(sum(duration)/3600,2)],
        "t_total_median": [round(statistics.median(duration)*len(duration),2), round(statistics.median(duration)*len(duration)/3600,2)],
        "sentences": len(data)
    }
    if verbose:
        logging.info(f"=============[ {stats['filename']} ]=============")
        logging.info(f"- Min time: {stats['t_min']} s")
        logging.info(f"- Mean time: {stats['t_mean']} s")
        logging.info(f"- Max time: {stats['t_max']} s")
        logging.info(f"{'-'*(30+len(stats['filename']))}")
        logging.info(f"- Total time (sum): {stats['t_total'][0]} s | {stats['t_total'][1]} h")
        logging.info(f"- Total sentences: {stats['sentences']}")
        logging.info(f"{'-'*(30+len(stats['filename']))}")
        logging.info(f"- Median time: {stats['t_median']} s")
        logging.info(f"- Total time (median): {stats['t_total_median'][0]} s | {stats['t_total_median'][1]} h")
        logging.info(f"{'='*(30+len(stats['filename']))}")
    if return_stats:
        return stats
    
def stats2xlsx(stats_list, dst_xlsx_filepath):
    """
    Export a list of statistics dictionaries to an Excel file.

    Parameters
    ----------
    stats_list : list of dict
        Each dictionary should contain the keys:
            - "filename"
            - "t_min"
            - "t_mean"
            - "t_max"
            - "t_total" (a list: [total_seconds, total_hours])
            - "sentences"
    
    dst_xlsx_filepath : str
        Path where the Excel (.xlsx) file will be saved.

    Returns
    -------
    None

    Notes
    -----
    - The Excel file will have a single sheet named "Stats".
    - Columns include: filename, min/mean/max times, total time in hours, and number of sentences.
    - Existing files at the destination path will be overwritten.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Stats"
    headers = ["filename", "t_min", "t_mean", "t_max", "t_total (h)", "sentences"]
    ws.append(headers)
    for stat in stats_list:
        row = [
            stat["filename"],
            stat["t_min"],
            stat["t_mean"],
            stat["t_max"],
            stat["t_total"][1],
            stat["sentences"]
        ]
        ws.append(row)
    wb.save(dst_xlsx_filepath)
    
def resultwer2xlsx(resultwer_list, dst_xlsx_filepath):
    """
    Export a list of WER (Word Error Rate) results to an Excel file.

    Parameters
    ----------
    resultwer_list : list of dict
        Each dictionary should contain the keys:
            - "filename"
            - "mean_wer_cp"
            - "mean_wer"
            - "total_wer_cp"
            - "total_wer"
    
    dst_xlsx_filepath : str
        Path where the Excel (.xlsx) file will be saved.

    Returns
    -------
    None

    Notes
    -----
    - The Excel file will have a single sheet named "WER Results".
    - Columns include: filename, mean WER with/without case-preserving, and total WER with/without case-preserving.
    - Existing files at the destination path will be overwritten.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "WER Results"
    headers = ["filename", "mean_wer_cp", "mean_wer", "total_wer_cp", "total_wer"]
    ws.append(headers)
    for resultwer in resultwer_list:
        row = [
            resultwer["filename"],
            resultwer["mean_wer_cp"],
            resultwer["mean_wer"],
            resultwer["total_wer_cp"],
            resultwer["total_wer"]
        ]
        ws.append(row)
    wb.save(dst_xlsx_filepath)