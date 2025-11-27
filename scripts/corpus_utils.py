import json
import os
import statistics
import openpyxl
import logging

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
    
def manifest_time_stats(manifest, return_stats: bool = False, verbose: bool = True):
    data = read_manifest(manifest)
    duration = [float(item['duration']) for item in data]
    stats = {
        "filename": os.path.split(manifest)[1],
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
        logging.info(f"\tMin time: {stats['t_min']} s")
        logging.info(f"\tMean time: {stats['t_mean']} s")
        logging.info(f"\tMax time: {stats['t_max']} s")
        logging.info(f"\n\tTotal time (sum): {stats['t_total'][0]} s | {stats['t_total'][1]} h")
        logging.info(f"\tTotal sentences: {stats['sentences']}")
        logging.info(f"\n\tMedian time: {stats['t_median']} s")
        logging.info(f"\tTotal time (median): {stats['t_total_median'][0]} s | {stats['t_total_median'][1]} h")
        logging.info(f"==============={'='*len(stats['filename'])}===============")
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