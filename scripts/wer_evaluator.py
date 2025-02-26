import jiwer
import corpus_utils as cu
from normalizer import TextNormalizer

def calculate_wer(data, normalizer = TextNormalizer, pred_text_tag: str="pred_text", manifest_name: str="sample.json", verbose: bool=True, return_wer: bool=False):
    """ 
    Calculate WER with and without C&P 
    Returns the original data with those 2 calculated WERs per item
    It also returns the total and mean WER for the dataset
    """
    normalizer_params = vars(normalizer).copy()
    normalizer_params["tag"] = pred_text_tag
    secondary_normalizer = TextNormalizer(**normalizer_params)
    wer_list = []
    wer_cp_list = []
    data_clean = [dict(item) for item in data]
    data_clean = normalizer.clean_sentences(data_clean)
    if secondary_normalizer:
        data_clean = secondary_normalizer.clean_sentences(data_clean)
    new_data = []
    total_reference = ""
    total_reference_clean = ""
    total_pred = ""
    for i,item in enumerate(data):
        item_clean = data_clean[i]
        # Calculate wer with C&P for each sentence
        wer_cp = jiwer.wer(item["text"], item["pred_text"])
        wer_cp_list.append(wer_cp)
        item['wer_cp'] = wer_cp
        # Calculate normalized wer for each sentence
        wer = jiwer.wer(item_clean["text"], item_clean["pred_text"])
        wer_list.append(wer)
        item['wer'] = wer
        new_data.append(item)
        total_reference += " " + item["text"]
        total_reference_clean += " " + item_clean["text"]
        total_pred += " " + item["pred_text"]
    total_wer_cp = jiwer.wer(total_reference.strip(), total_pred.strip())
    total_wer = jiwer.wer(total_reference_clean.strip(), total_pred.strip())
    mean_wer_cp = sum(wer_cp_list)/len(wer_cp_list)   
    mean_wer = sum(wer_list)/len(wer_list)
    if verbose:
        print(f"=============[ {manifest_name} ]=============")
        print(f"\t Mean WER C&P: {round(mean_wer_cp*100,2)} %")
        print(f"\t     Mean WER: {round(mean_wer*100,2)} %")
        print(f"\tTotal WER C&P: {round(total_wer_cp*100,2)} %")
        print(f"\t    Total WER: {round(total_wer*100,2)} %")
        print(f"==============={'='*len(manifest_name)}===============")
    if return_wer: 
        return new_data, mean_wer_cp, mean_wer, total_wer_cp, total_wer