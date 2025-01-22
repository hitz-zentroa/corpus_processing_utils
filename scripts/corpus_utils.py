import json

def read_manifest(manifest_filepath):
    # Reads a .json with dict items in each line
    # Returns a list with all dicts
    print("Reading:",manifest_filepath)
    data = []
    try:
        f = open(manifest_filepath, 'r', encoding='utf-8')
    except:
        raise Exception(f"Manifest file could not be opened: {manifest_filepath}")
    for line in f:
        item = json.loads(line)
        data.append(item)
    f.close()
    return data

def write_manifest(manifest_filepath, data, ensure_ascii: bool = False, return_manifest_filepath: bool = False):
    # Writes a list of dicts in a .json file
    f = open(manifest_filepath, "w", encoding="utf-8")
    for item in data:
        f.write(json.dumps(item,ensure_ascii=ensure_ascii) + "\n")
    f.close()
    print("End Writing manifest:", manifest_filepath)
    if return_manifest_filepath:
        return manifest_filepath
    else:
        return None