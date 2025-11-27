import json
import os
import statistics
import openpyxl
import logging
import pandas as pd
from tqdm import tqdm
import soundfile as sf

def read_manifest(manifest_filepath, verbose: bool = True):
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

def tsv2data(tsv_filepath: str, clips_folder: str="", audio_field="path", text_field="sentence", duration_field=None, header='infer'):
    """
    Reads a TSV file containing audio paths and text sentences and returns a structured dataset.
    Each row in the TSV represents a single audio-text pair. Optionally, a column for
    precomputed durations can be provided; otherwise, the function calculates durations
    from the audio files using `soundfile`.
    Parameters
    ----------
    tsv_filepath : str
        Path to the TSV file to read.
    clips_folder : str, optional
        Base folder to prepend to audio file paths (default is "").
    audio_field : str or int, optional
        Column containing audio file paths. Default is "path".  
        Can be an integer index if the TSV has no headers.
    text_field : str or int, optional
        Column containing text sentences. Default is "sentence".  
        Can be an integer index if the TSV has no headers.
    duration_field : str or int, optional
        Column containing duration values. If empty, durations are ignored.
        Can be an integer index if the TSV has no headers.
    header : int, list of int, 'infer', or None, optional
        Row(s) to use as the column names. Default is 'infer', which uses the first row as header.
        Set to None if the TSV has no headers.
    """
    data=[]
    df = pd.read_csv(tsv_filepath, sep='\t', header=header)
    for idx in tqdm(range(len(df))):
        audio_filepath = os.path.join(clips_folder,df[audio_field][idx])
        text = df[text_field][idx]
        if duration_field:
            try:
                duration = df[duration_field][idx]
            except:
                raise Exception(f"Audio file could not be opened: {audio_filepath}")
        else:
            f = sf.SoundFile(audio_filepath)
            duration = len(f) / f.samplerate
        item = {
            'audio_filepath': audio_filepath,
            'text': text,
            'duration': duration
        }
        data.append(item)
    return data
    
def pairedfiles2data(clips_folder, sentences_folder):
    """
    Creates a structured dataset by pairing audio files with their corresponding text sentences.
    Each `.txt` file in `sentences_folder` is expected to have a corresponding `.wav`
    file in `clips_folder` with the same base filename.

    Parameters
    ----------
    clips_folder : str
        Path to the folder containing audio files (.wav).
    sentences_folder : str
        Path to the folder containing sentence files (.txt).
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
    hashed_sentences = [hash(item["text"]) for item in tqdm(data)]
    return hashed_sentences

def reduce_data(data, compare_data=None, hashed_data=None, hashed_compare=None):
    """
    Reduce a dataset by removing duplicates or items present in another dataset.
    
    Parameters
    ----------
    data : list
        Dataset to reduce.
    compare_data : list, optional
        If provided, remove all items from `data` that are in `compare_data`.
        If None, remove duplicates within `data`.
    hashed_data : list of int, optional
        Precomputed hashes of `data`.
    hashed_compare : list of int, optional
        Precomputed hashes of `compare_data`.
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
    
def reduce_common_voice(path):
    validated_tsv = f"{path}/validated.tsv"
    test_tsv = f"{path}/validated.tsv"
    dev_tsv = f"{path}/validated.tsv"
    
    clips_folder = f"{path}/clips"
    manifests_path = f"{path}/manifests"
    os.makedirs(manifests_path, exist_ok=True)
    
    for tsv in [validated_tsv, test_tsv, dev_tsv]:
        name = os.path.split(tsv)[1]
        manifest_filepath = f"{manifests_path}/{name.replace('.tsv','.json')}"
        data = tsv2data(tsv_filepath=tsv, clips_folder=clips_folder, audio_filed="path", text_field="sentence")
        write_manifest(manifest_filepath, data)