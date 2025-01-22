import json
import os
import statistics

def read_manifest(manifest_filepath):
    print("Reading:",manifest_filepath)
    try:
        f = open(manifest_filepath, 'r', encoding='utf-8')
        data = [json.loads(line) for line in f]
        f.close()
        return data
    except:
        raise Exception(f"Manifest file could not be opened: {manifest_filepath}")

def write_manifest(manifest_filepath, data, ensure_ascii: bool = False, return_manifest_filepath: bool = False):
    f = open(manifest_filepath, "w", encoding="utf-8")
    for item in data:
        f.write(json.dumps(item,ensure_ascii=ensure_ascii) + "\n")
    f.close()
    print("End Writing manifest:", manifest_filepath)
    if return_manifest_filepath:
        return manifest_filepath
    else:
        return None
    
def manifest_time_stats(manifest):
    data = read_manifest(manifest)
    duration = [float(item["duration"]) for item in data]
    filename = os.path.split(manifest)[1]
    print(f"=============[ {filename} ]=============")
    print("\tMin time: ",round(min(duration),2), "s")
    print("\tMean time:",round(statistics.mean(duration),2), "s")
    print("\tMax time: ",round(max(duration),2), "s")
    print("\n\tTotal time (sum):",round(sum(duration),2), "s |",round(sum(duration)/3600,2), 'h')
    print("\tTotal sentences: ",len(data))
    print("\n\tMedian time:",round(statistics.median(duration),2), "s")
    print("\tTotal time (median):",round(statistics.median(duration)*len(duration),2), "s |",round(statistics.median(duration)*len(duration)/3600,2), 'h')
    print(f"==============={'='*len(filename)}===============")