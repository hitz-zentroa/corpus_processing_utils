import json
import pandas as pd
import os
import re
import statistics
import shutil
from tqdm import tqdm
import jiwer
from datasets import load_dataset
import copy
import sox
from pydub import AudioSegment
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

#############################################################

def print_separator(char='#'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(char*100)
            print(f"Executing: {func.__name__}")
            result = func(*args, **kwargs)
            print(char*100 + "\n")
            return result
        return wrapper
    return decorator

#############################################################

def read_manifest(manifest_filepath):
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

def read_txt(txt_filepath):
    data=[]
    try:
        f = open(txt_filepath,'r',encoding='utf-8')
    except:
        raise Exception(f"File could not be opened: {txt_filepath}")
    for line in f:
        item = {'text': line.strip()}
        data.append(item)
    f.close()
    return data

def write_manifest(manifest_filepath, data, ensure_ascii: bool = False):
    f = open(manifest_filepath, "w", encoding="utf-8")
    for item in data:
        f.write(json.dumps(item,ensure_ascii=ensure_ascii) + "\n")
    f.close()
    print("End Writing manifest:", manifest_filepath)

@print_separator("#")
def tsv2data(tsv_filepath,clips_folder: str=""):
    data=[]
    df = pd.read_csv(tsv_filepath, sep='\t')
    for idx in tqdm(range(len(df))):
        audio_filepath = clips_folder + "/" + df['file_name'][idx]
        item = {
            'audio_filepath': audio_filepath,
            'text': df['transcription'][idx],
            # 'client_id': df['client_id'][idx],
            'duration': df['duration'][idx]
            # 'up_votes': int(df['up_votes'].fillna(0)[idx]),
            # 'down_votes': int(df['down_votes'].fillna(0)[idx]),
            # 'age': df['age'].fillna(pd.NaT)[idx], 
            # 'gender': df['gender'].fillna(pd.NaT)[idx],
            # 'accents': df['accents'].fillna(pd.NaT)[idx],
            # 'variant': df['accents'].fillna(pd.NaT)[idx],
            # 'locale': df['accents'].fillna(pd.NaT)[idx],            
        }
        data.append(item)
    return data

@print_separator("#")
def decode_clips(data, destination_folder, samplerate: int = 16000):
    decoded_data = []
    tfm = sox.Transformer()
    for item in tqdm(data):
        decoded_item = dict(item)
        _, audio_file = os.path.split(item['audio_filepath'])
        audio_name, audio_ext = os.path.splitext(audio_file)
        # if not (audio_ext == ".wav" or sox.file_info.sample_rate(item["audio_filepath"])==samplerate):
        if not (audio_ext == ".wav"):
            decoded_item['audio_filepath'] = destination_folder + "/" + audio_name + '.wav'
            if not os.path.exists(decoded_item['audio_filepath']):
                tfm.set_output_format(file_type="wav",rate=samplerate, channels=1)
                tfm.build(input_filepath=item['audio_filepath'],output_filepath=decoded_item['audio_filepath'])
        decoded_item['duration'] = sox.file_info.duration(decoded_item['audio_filepath'])
        decoded_data.append(decoded_item)
    return decoded_data

@print_separator("#")
def convert2mp3(data,output_dir):
    #This function is very specefic, modify it where its needed when you use it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for item in tqdm(data):
        audio_path,audio_name = os.path.split(item["audio_filepath"])
        _,speaker_folder = os.path.split(audio_path)
        new_output_dir = os.path.join(output_dir,speaker_folder)
        if not os.path.exists(new_output_dir):
            os.mkdir(new_output_dir)
        output_filepath = os.path.join(new_output_dir,audio_name.replace(".wav",".mp3"))
        audio = AudioSegment.from_wav(item["audio_filepath"])
        audio = audio.set_frame_rate(16000)
        audio.export(output_filepath,format="mp3")

@print_separator("#")
def pairedfiles2data(clips_folder, sentences_folder):
    data=[]
    for file in os.listdir(sentences_folder):
        if file.endswith(".txt"):
            audio_filepath = os.path.join(clips_folder,file.replace(".txt",".wav"))
            try:
                duration = sox.file_info.duration(audio_filepath)
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

def remove_special_chars(item, cp, tag):
    chars_to_ignore = "[\:\-‑;()«»…\]\[/\*–‽+\%&_\\½√><|€™$•¼}º{~—=“\"”″‟„’'‘`ʽ']\n\x01\x03"
    # chars_to_ignore = "[\:\-‑;()«»…\]\[/\*–‽+\%&_\\½√><|€™$•¼}º{~—=“\"”″‟„’'‘`ʽ']\n\x01\x031234567890"
    if not cp:
        chars_to_ignore = chars_to_ignore + "\.\,\?\¿\¡\!"
        item[tag] = item[tag].lower()
    item[tag] = re.sub(rf"[{re.escape(chars_to_ignore)}]", " ", item[tag]) # replace chars by space and lower all
    item[tag] = re.sub(r" +", " ", item[tag]).strip() # merge multiple spaces and erase last space
    return item

def replace_diacritics(item, lang, tag):
    if lang == "eu":
        item[tag] = re.sub(r"[éèëēê]", "e", item[tag])
        item[tag] = re.sub(r"[ãâāáàä]", "a", item[tag])
        item[tag] = re.sub(r"[úùūüû]", "u", item[tag])
        item[tag] = re.sub(r"[ôōóòöõ]", "o", item[tag])
        item[tag] = re.sub(r"[ćç]", "c", item[tag])
        item[tag] = re.sub(r"[ïīíìî]", "i", item[tag])
    elif lang == "es":
        item[tag] = re.sub(r"[èëēê]", "e", item[tag])
        item[tag] = re.sub(r"[ãâāàä]", "a", item[tag])
        item[tag] = re.sub(r"[ùūû]", "u", item[tag])
        item[tag] = re.sub(r"[ôōòöõ]", "o", item[tag])
        item[tag] = re.sub(r"[ćç]", "c", item[tag])
        item[tag] = re.sub(r"[ïīìî]", "i", item[tag])
    else:
        print("ERROR: Wrong Language, only supported: EU or ES")
    return item

@print_separator(".")
def clean_sentences(data, lang, cp: bool = False, tag: str="text"):
    lang.lower()
    clean_data = []
    unclean_char_list = set()
    clean_char_list = set()
    for item in data:
        unclean_char_list.update(set(item[tag]))
        clean_item = remove_special_chars(item,cp,tag)
        clean_item = replace_diacritics(clean_item,lang,tag)
        clean_data.append(clean_item)
        clean_char_list.update(set(clean_item[tag]))
    print(f"\nData character list before cleaning: Size = {len(unclean_char_list)}\n {sorted(unclean_char_list)}")
    print(f"\nData character list after cleaning: Size = {len(clean_char_list)}\n {sorted(clean_char_list)}")
    return clean_data

def hash_sentences(data):
    hashed_sentences = [hash(item["text"]) for item in tqdm(data)]
    return hashed_sentences

@print_separator("-")
def reduce_test_data(test_data, hashed_test: list = None):
    # Returns the same data with no repeated phrases.
    if hashed_test is None:
        hashed_test = hash_sentences(test_data)
    test_data_reduced = []
    for i,item in tqdm(enumerate(test_data)):
        if hashed_test[i] not in hashed_test[:i]:
            test_data_reduced.append(item)
    return test_data_reduced

@print_separator("+")
def reduce_target_data(target_data, test_data, hashed_target: list = None, hashed_test: list = None):
    # Returns a data with all the phrases of target_data that are NOT in test_data "cleans target_data from phrases of test_data"
    if hashed_target is None:
        hashed_target = hash_sentences(target_data)
    if hashed_test is None:
        hashed_test = hash_sentences(test_data)
    target_data_reduced = []
    hashed_test_set = set(hashed_test)
    for i,item in tqdm(enumerate(target_data)):
        if hashed_target[i] not in hashed_test_set:
            target_data_reduced.append(item)
    return target_data_reduced

def manifest_time_stats(manifest):
    data = read_manifest(manifest)
    duration = []
    for i,item in enumerate(data):
        duration.append(float(item["duration"]))
    print(f"=============[ {os.path.split(manifest)[1]} ]=============")
    print("\tMin time: ",round(min(duration),2), "s")
    print("\tMean time:",round(statistics.mean(duration),2), "s")
    print("\tMax time: ",round(max(duration),2), "s")
    print("\n\tTotal time (sum):",round(sum(duration),2), "s |",round(sum(duration)/3600,2), 'h')
    print("\tTotal sentences: ",len(data))
    print("\n\tMedian time:",round(statistics.median(duration),2), "s")
    print("\tTotal time (median):",round(statistics.median(duration)*i,2), "s |",round(statistics.median(duration)*i/3600,2), 'h')
    print("===========================================================")

@print_separator("·")
def evaluate_leak(target_data, test_data, target_name: str = "target_manifest", test_name: str = "test_manifest", print_phrases_idx: bool = False, hashed_target: list = None, hashed_test: list = None):        
    # Evaluates how much sentences of the test data are in the target data
    if hashed_target is None:
        hashed_target = hash_sentences(target_data)
    if hashed_test is None:
        hashed_test = hash_sentences(test_data)
    
    phrases_idx=[]    
    leaked_phrases = []
    hashed_target_set = set(hashed_target)
    for i,item in tqdm(enumerate(test_data)):
        if hashed_test[i] in hashed_target_set:
            leaked_phrases.append(item["text"])
            phrases_idx.append(i+1)
    print(f"\tLeak en << {target_name} >>:")
    len_leaked_phrases = len(leaked_phrases)
    len_test_data = len(test_data)
    print(f"\t- << {test_name} >> = {round((len_leaked_phrases/len_test_data)*100,2)}% ({len_leaked_phrases}/{len_test_data})")
    if print_phrases_idx:
        print(phrases_idx)

def download_dataset(language, split):
    # Funcion especifica para descargar dataset de https://huggingface.co/datasets/gttsehu/basque_parliament_1
    files_path = os.path.join("/mnt/corpus/basque_parliament_1_gttsehu",language)
    manifest_filepath = os.path.join(files_path,split+".json")
    ds = load_dataset("gttsehu/basque_parliament_1", language, split=split)
    data = []
    for line in tqdm(ds):
        src_audio_filepath = line["path"]
        _, audio_name = os.path.split(src_audio_filepath)
        dst_audio_filepath = os.path.join(files_path,"clips",split,audio_name)
        if not os.path.exists(dst_audio_filepath):
            shutil.move(src_audio_filepath,dst_audio_filepath)
        item = {
            "audio_filepath": dst_audio_filepath,
            "text": line["sentence"],
            "client_id": line["speaker_id"],
            "lang": line["language"],
            "duration": line["length"]
        }
        data.append(item)
    write_manifest(manifest_filepath, data)

def calculate_wer(item, normalizer):
    reference = item["text"]
    hypothesis = item["pred_text"]
    # reference = normalizer(item["text"]).strip()
    # hypothesis = normalizer(item["pred_text"]).strip()
    wer = jiwer.wer(reference, hypothesis)
    return wer

def print_result(item,wer):
    print(" Reference: ",item["text"])
    print("Hypothesis: ",item["pred_text"])
    print("  Pred_WER: ",item["wer"])
    print("  Calc_WER: ",wer)
    print("---------------------")

def evaluate_wer(data, manifest_name: str="sample_file"):
    # Calculate WER with an without C&P and returns the orgiinal data with those 2 wers per item
    wer_list = []
    wer_cp_list = []
    data_clean = copy.deepcopy(data)
    data_clean = clean_sentences(data_clean,'eu',tag='text')
    data_clean = clean_sentences(data_clean,'eu',tag='pred_text')
    new_data = []
    normalizer = BasicTextNormalizer()
    for i,item in enumerate(data):
        item_clean = data_clean[i]
        # Calculate and write wer with C&P for each sentence
        wer_cp = calculate_wer(item,normalizer)
        wer_cp_list.append(wer_cp)
        item['wer_cp'] = wer_cp
        # Calculate and write normalized wer for each sentence
        wer = calculate_wer(item_clean,normalizer)
        wer_list.append(wer)
        item['wer'] = wer
        new_data.append(item)
    wer_cp_total = sum(wer_cp_list)/len(wer_cp_list)   
    wer_total = sum(wer_list)/len(wer_list)
    print("Manifest:", manifest_name)
    print(f"Total_WER_C&P: {round(wer_cp_total*100,2)} %")
    print(f"Total_WER: {round(wer_total*100,2)} %")
    return new_data

def simple_wer(data, manifest_name):
    # Prints the total wer and wer_cp of a file with those tags for each item
    wer_list = []
    wer_cp_list = []
    for item in data:
        wer_cp_list.append(item['wer_cp'])
        wer_list.append(item['wer'])
    wer_cp_total = sum(wer_cp_list)/len(wer_cp_list)   
    wer_total = sum(wer_list)/len(wer_list)
    print("Manifest:", manifest_name)
    print(f"Total_WER_C&P: {round(wer_cp_total*100,2)} %")
    print(f"Total_WER: {round(wer_total*100,2)} %\n")
    return wer_cp_total, wer_total

def compute_wer(path):
    files = os.listdir(path)
    for manifest in files:
        if manifest.endswith(".json"):
            data = read_manifest(os.path.join(path,manifest))
            # simple_wer(data,manifest)
            new_data = evaluate_wer(data, manifest)
            write_manifest(os.path.join(path,manifest),new_data)

# FUNCIONES PENDIENTES
# split_dataset: Return 2 datasets, train / test using _var%.


#############################################################
def main_leakSearch():
    path = "/mnt/aholab/asierhv/ASR_eu_test_files"

    # target_txt = os.path.join(path,"LM_files/wiki.txt")
    target_txt = os.path.join(path,"LM_files/corpora.txt")

    test_manifests = list()
    test_manifests.append(os.path.join(path,"test_CORPUS/CV16_test_processed.json"))
    test_manifests.append(os.path.join(path,"test_CORPUS/banco_voces_corpus_processed.json"))
    test_manifests.append(os.path.join(path,"test_CORPUS/test_openSLR_processed_reduced.json"))
    test_manifests.append(os.path.join(path,"test_CORPUS/gtts_parliament_eu_test_decoded_processed.json"))
    test_manifests.append(os.path.join(path,"test_CORPUS/gtts_parliament_eu_validation_decoded_processed.json"))
    
    target_data = read_txt(target_txt)

    for test_manifest in test_manifests:
        test_data = read_manifest(test_manifest)
        evaluate_leak(target_data=target_data,test_data=test_data,target_name=os.path.split(target_txt)[1],test_name=os.path.split(test_manifest)[1])

def main_datasetDownload():
    language = "es" # es, eu, bi
    splits = ["train_clean","validation","test"]
    for split in splits:
        download_dataset(language,split)

def main_convert2mp3():
    manifest = "/scratch/asierhv/manifests/composite_corpus_eu/json_mp3/localtest_ahomytts.json"
    output_dir = "/scratch/asierhv/composite_corpus_eu/data/localtest_ahomytts"
    data = read_manifest(manifest)
    convert2mp3(data,output_dir)

def main():
    model_size_list = ["tiny","base","small","medium","large","large-v2"]
    for model_size in model_size_list:
        # model = f"whisper-{model_size}-eu-composite_corpus_eu"
        model = f"zuazo-whisper-{model_size}-eu-cv16_1"
        
        lm = "no_LM"
        # lm = "with_LM"

        path = "/scratch/asierhv/results/predictions/"+model+"_predictions/"+lm
        # path = "/scratch/asierhv/results/predictions/mp3/"+model+"_predictions/"+lm
        
        # path = "/scratch/asierhv/results/predictions/pruebas"
        
        # compute_wer(path)
        files = os.listdir(path)
        wer_summary_list = []
        for manifest in files:
            if manifest.endswith(".json"):
                data = read_manifest(os.path.join(path,manifest))
                wer_cp, wer = simple_wer(data,manifest)
                wer_summary = {"model": model,"manifest": manifest,"wer_cp": wer_cp, "wer": wer}
                wer_summary_list.append(wer_summary)
                # new_data = evaluate_wer(data, manifest)
                # write_manifest(os.path.join(path,manifest),new_data)
        write_manifest(os.path.join(path,"wer_summary.json"),wer_summary_list)
            
if __name__== "__main__":
    # compute_wer("/scratch/asierhv/results/predictions/pruebas")
    main()