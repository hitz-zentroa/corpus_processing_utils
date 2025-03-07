import re
from tqdm import tqdm
import corpus_utils as cu

class TextNormalizer:
    def __init__(self, lang: str, tag: str = "text", keep_cp: bool = False, 
                 remove_acronyms: bool = False, blacklist_terms = None, 
                 min_duration: float = 0.025, max_duration: float = 240,
                 verbose: bool = True):
        """
        Initializes the text cleaner with the necessary parameters.
        :param lang: Language ('es' or 'eu') if Bilingual 'es+eu' is wanted just select 'es'.
        :param tag: Key field for the text in the data.
        :param keep_cp: Whether to preserve Capitalization and Punctuation.
        :param remove_acronyms: Will remove the sentences with acronyms if True
        :param blacklist_terms: List of terms to remove (if provided).
        :param min/max_duration: duration threshold in seconds for the audios, will remove the sentence if it's out of bounds.
        :param verbose: for logging info.
        """
        if lang not in ['es', 'eu']:
            raise ValueError(f"ERROR: Language '{lang}' NOT Supported.\n Supported languages:\n\t- Spanish: 'es'.\n\t- Basque: 'eu'")
        self.lang = lang.lower()
        self.tag = tag
        self.keep_cp = keep_cp
        self.unclean_char_list = set()
        self.clean_char_list = set()
        self.remove_acronyms = remove_acronyms
        self.blacklist_terms = blacklist_terms
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.verbose = verbose

    def replace_diacritics(self, item):
        """Replaces diacritic characters with their normalized versions."""
        diacritic_map = {
            r"[Æ]": "Ae", r"[Œ]": "Oe", r"[Ж]": "Zh", r"[Х]": "H", r"[Щ]": "Shch", r"[Ш]": "Sh", r"[Ф]": "F",
            r"[Ч]": "Ch", r"[Ц]": "Ts", r"[Þ]": "Th", r"[Α]": "A", r"[Β]": "V", r"[Γ]": "G", r"[Δ]": "D",
            r"[Ζ]": "Z", r"[Η]": "I", r"[Θ]": "Th", r"[Κ]": "K", r"[Λ]": "L", r"[Μ]": "M", r"[Ν]": "N",
            r"[Ξ]": "Ks", r"[Π]": "P", r"[Ρ]": "R", r"[Σ]": "S", r"[Τ]": "T", r"[Υ]": "I", r"[Φ]": "F",
            r"[Χ]": "J", r"[Ψ]": "Ps", r"[Ω]": "O", r"[ß]": "ss", r"[ж]": "zh", r"[х]": "h", r"[щ]": "shch",
            r"[ш]": "sh", r"[ф]": "f", r"[ч]": "ch", r"[ц]": "ts", r"[ð]": "d", r"[ђ]": "dj", r"[α]": "a",
            r"[β]": "v", r"[γ]": "g", r"[δ]": "d", r"[ζ]": "z", r"[η]": "i", r"[θ]": "th", r"[κ]": "k",
            r"[λ]": "l", r"[μ]": "m", r"[ν]": "n", r"[ξ]": "ks", r"[π]": "p", r"[ρ]": "r", r"[σς]": "s",
            r"[υ]": "u", r"[φ]": "f", r"[χ]": "j", r"[ψ]": "ps", r"[ÈËÊЕЭ]": "E", r"[АÃÂÀÄÅ]": "A",
            r"[ÙÛŪ]": "U", r"[ÔÖÒÕØΟ]": "O", r"[ÇĆČ]": "C", r"[ÏÌÎĪ]": "I", r"[ÑŃǸ]": "Ñ", r"[ÝŶŸ]": "Y",
            r"[èëēêе]": "e", r"[аãâāàä]": "a", r"[ùūû]": "u", r"[ôōòöõ]": "o", r"[ćç]": "c", r"[ïīìî]": "i",
            r"[ż]": "z", r"[ ]": " "
        }
        for pattern, replacement in diacritic_map.items():
            item[self.tag] = re.sub(pattern, replacement, item[self.tag])
        if self.lang == "eu":
            eu_specific = {
                r"[É]": "E", r"[Á]": "A", r"[ÚÜ]": "U", r"[Ó]": "O", r"[Í]": "I",
                r"[é]": "e", r"[á]": "a", r"[úü]": "u", r"[ó]": "o", r"[í]": "i"
            }
            for pattern, replacement in eu_specific.items():
                item[self.tag] = re.sub(pattern, replacement, item[self.tag])
        return item

    def remove_special_chars_whitelist(self, item):
        """Removes not allowed special characters."""
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúüÁÉÍÓÚÜñÑ "
        if self.keep_cp:
            allowed_chars += ".,¿?¡!;:"
        allowed_chars_pattern = f"[^{allowed_chars}]"
        if self.blacklist_terms:
            for term in self.blacklist_terms:
                item[self.tag] = re.sub(term, "", item[self.tag], flags=re.IGNORECASE)
        item[self.tag] = re.sub(allowed_chars_pattern, " ", item[self.tag])
        item[self.tag] = re.sub(r" +", " ", item[self.tag]).strip()
        if not self.keep_cp:
            item[self.tag] = item[self.tag].lower()
        return item
    
    def in_duration_threshold(self, item):
        """Removes sentences out of min-max threshold"""
        if item["duration"] < self.min_duration or item["duration"] > self.max_duration:
            if self.verbose: print("Removed (duration out of bounds):", item["audio_filepath"])
            return False
        else:
            return True

    def clean_sentences(self, data):
        clean_data = []
        n = 0
        m = 0
        for item in tqdm(data):
            if self.in_duration_threshold(item):
                acronyms = bool(re.search(r'\b[\w\d]*[A-Z]{2,}[\w\d]*\b', item[self.tag])) if self.remove_acronyms else False
                if not acronyms:
                    self.unclean_char_list.update(set(item[self.tag]))
                    item = self.replace_diacritics(item)
                    item = self.remove_special_chars_whitelist(item)
                    if re.search(r"[A-Za-z]",item["text"]):
                        clean_data.append(item)
                    else:
                        m += 1
                        if self.verbose: print("Removed (no text on sentence):", item["audio_filepath"])
                    self.clean_char_list.update(set(item[self.tag]))
                else:
                    n += 1
                    if self.verbose: print("Sentence with acronyms:", item[self.tag])
        if self.verbose:
            print(f"\nCharacter list before cleaning: Size = {len(self.unclean_char_list)}\n {sorted(self.unclean_char_list)}")
            print(f"\nCharacter list after cleaning: Size = {len(self.clean_char_list)}\n {sorted(self.clean_char_list)}")
            if self.remove_acronyms: print(f"\nSentences with acronyms eliminated: {n}/{len(data)} ({round(n / len(data), 2) * 100}%)")
            print(f"\nTotal sentences eliminated: {n+m}/{len(data)} ({round(n+m / len(data), 2) * 100}%)")
        return clean_data

def main():
    blacklist_terms =[
        "\(inint\)", "\(inint\(", "\(Inint\)", "\(init\)",
        "\(gabe\)","\(Many speakers\)",
        "\(Ri\)","\(RI\)","\(RU\)",
        "\(MU\)","\(LL\)","\(BO\)","\-c\}","\-n\}"
    ]
    json = "example_eu.json"
    data = cu.read_manifest(f"./manifests/{json}")
    eu_normalizer = TextNormalizer(lang='eu', keep_cp=False, blacklist_terms=blacklist_terms)
    clean_data = eu_normalizer.clean_sentences(data)
    json_clean=json.replace(".json","_clean.json")
    cu.write_manifest(f"./manifests/processed/{json_clean}", clean_data)

if __name__=="__main__":
    main()
