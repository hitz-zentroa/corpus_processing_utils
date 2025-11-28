import jiwer
import os
import logging
from normalizer import TextNormalizer
import corpus_utils as cu

def calculate_wer(manifest_filepath, lang: str="es",
                  text_tag: str="text", cp_text_tag: str="cp_text",
                  pred_text_tag: str="pred_text", cp_pred_text_tag: str="cp_pred_text",
                  cp_field: bool=False, return_wer: bool=False, verbose: bool=True):
    """
    Calculate sentence-level and corpus-level Word Error Rate (WER) from a manifest file.

    Parameters
    ----------
    manifest_filepath : str
        Path to the manifest file (JSONL format). Each entry must contain 
        reference text and predicted text fields.

    lang : str, optional (default="es")
        Language code used for text normalization.

    text_tag : str, optional (default="text")
        Key in the manifest representing the reference text.

    cp_text_tag : str, optional (default="cp_text")
        Key representing case-preserved reference text. Used only when `cp_field=True`.

    pred_text_tag : str, optional (default="pred_text")
        Key representing the predicted text.

    cp_pred_text_tag : str, optional (default="cp_pred_text")
        Key representing case-preserved predicted text. Used only when `cp_field=True`.

    cp_field : bool, optional (default=False)
        If True, also computes WER using case-preserved (C&P) text fields.

    return_wer : bool, optional (default=False)
        If True, returns both the cleaned manifest entries and the result dictionary.

    verbose : bool, optional (default=True)
        If True, logs mean and total WER values to the console.

    Returns
    -------
    tuple (list of dict, dict), optional
        Returned only if `return_wer=True`.

        - data_clean : list of dict  
            Manifest entries after text normalization and per-sentence WER annotation.  
            Keys added:
                * "wer"  
                * "wer_cp" (only if cp_field=True)

        - result : dict  
            Summary WER statistics containing:
                * "filename"  
                * "mean_wer"  
                * "total_wer"  
                * "mean_wer_cp" (None if cp_field=False)  
                * "total_wer_cp" (None if cp_field=False)

    Notes
    -----
    - Text normalization is performed using `TextNormalizer`, once for reference text and once for predictions.
    - Per-sentence WER is computed using `jiwer.wer()`.
    - Corpus-level WER is computed by concatenating all texts and evaluating on the full strings.
    - Output values are raw WER scores (0.0–1.0), not percentages.
    - Log output shows percentages for readability.
    """
    filename = (os.path.split(manifest_filepath)[1]).replace(".json","")

    data = cu.read_manifest(manifest_filepath, verbose=False)
    data_clean = [dict(item) for item in data] 

    # Create the normalizers for each field
    normalizer = TextNormalizer(lang=lang, tag=text_tag, verbose=False)
    pred_normalizer = TextNormalizer(lang=lang, tag=pred_text_tag, verbose=False)
    data_clean = normalizer(data_clean)
    data_clean = pred_normalizer(data_clean)
    if cp_field:
        cp_normalizer = TextNormalizer(lang=lang, tag=cp_text_tag, keep_cp=True, verbose=False)
        cp_pred_normalizer = TextNormalizer(lang=lang, tag=cp_pred_text_tag, keep_cp=True, verbose=False)
        data_clean = cp_normalizer(data_clean)
        data_clean = cp_pred_normalizer(data_clean)

    wer_list = []
    wer_cp_list = []
    total_text = ""
    total_cp_text = ""
    total_pred_text = ""
    total_cp_pred_text = ""
    for item in data_clean:
        # Calculate normalized wer for each sentence
        wer = jiwer.wer(item["text"], item["pred_text"])
        wer_list.append(wer)
        item['wer'] = wer
        total_text += " " + item["text"]
        total_pred_text += " " + item["pred_text"]

        if cp_field:
            # Calculate wer with C&P for each sentence
            wer_cp = jiwer.wer(item["cp_text"], item["cp_pred_text"])
            wer_cp_list.append(wer_cp)
            item['wer_cp'] = wer_cp
            total_cp_text += " " + item["cp_text"]
            total_cp_pred_text += " " + item["cp_pred_text"]

    total_wer = jiwer.wer(total_text.strip(), total_pred_text.strip())   
    mean_wer = sum(wer_list)/len(wer_list)
    if cp_field:
        total_wer_cp = jiwer.wer(total_cp_text.strip(), total_cp_pred_text.strip())
        mean_wer_cp = sum(wer_cp_list)/len(wer_cp_list)
    else:
        total_wer_cp = None
        mean_wer_cp = None

    result = {
        "filename": filename,
        "mean_wer_cp": mean_wer_cp,
        "mean_wer": mean_wer,
        "total_wer_cp": total_wer_cp,
        "total_wer": total_wer
        }
    if verbose:
        logging.info(f"=============[ {result['filename']} ]=============")
        logging.info(f"- Mean WER: {round(result['mean_wer']*100,2)} %")
        logging.info(f"- Total WER: {round(result['total_wer']*100,2)} %")
        if cp_field:
            logging.info(f"{'-'*(30+len(result['filename']))}") 
            logging.info(f"- Mean WER C&P: {round(result['mean_wer_cp']*100,2)} %")
            logging.info(f"- Total WER C&P: {round(result['total_wer_cp']*100,2)} %")
        logging.info(f"{'='*(30+len(result['filename']))}")    
    if return_wer: 
        return data_clean, result