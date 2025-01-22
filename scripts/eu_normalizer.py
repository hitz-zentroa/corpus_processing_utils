import re
from tqdm import tqdm
import corpus_utils as cu

class TextNormalizer:
    def __init__(self, lang: str, tag: str = "text", cp: bool = False):
        """
        Initializes the text cleaner with the necessary parameters.
        :param lang: Language ('es' or 'eu').
        :param tag: Key field for the text in the data.
        :param cp: Whether to preserve Capitalization and Punctuation.
        """
        if lang not in ['es', 'eu']:
            raise ValueError(f"ERROR: Language '{lang}' NOT Supported.\n Supported languages:\n\t- Spanish: 'es'.\n\t- Basque: 'eu'")
        self.lang = lang.lower()
        self.tag = tag
        self.cp = cp
        self.unclean_char_list = set()
        self.clean_char_list = set()

    def check_acronyms(self, item):
        """Checks if the sentence contains acronyms (More than two consecutive capitals in a word)."""
        return bool(re.search(r'\b[\w\d]*[A-Z]{2,}[\w\d]*\b', item[self.tag]))

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
            r"[èëēêе]": "e", r"[аãâāàä]": "a", r"[ùūû]": "u", r"[ôōòöõ]": "o", r"[ćç]": "c", r"[ïīìî]": "i"
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
        """Removes disallowed special characters."""
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúüÁÉÍÓÚÜñÑ "
        if self.cp:
            allowed_chars += r"\.\,\?\¿\¡\!\;\:"

        allowed_chars_pattern = f"[^{re.escape(allowed_chars)}]"
        item[self.tag] = re.sub(allowed_chars_pattern, " ", item[self.tag])
        item[self.tag] = re.sub(r" +", " ", item[self.tag]).strip()
        if not self.cp:
            item[self.tag] = item[self.tag].lower()
        return item

    def clean_sentences(self, data):
        clean_data = []
        n = 0
        for item in tqdm(data):
            if not self.check_acronyms(item):
                self.unclean_char_list.update(set(item[self.tag]))
                item = self.replace_diacritics(item)
                item = self.remove_special_chars_whitelist(item)
                clean_data.append(item)
                self.clean_char_list.update(set(item[self.tag]))
            else:
                n += 1
                print("Sentence with acronyms:", item[self.tag])
        print(f"\nCharacter list before cleaning: Size = {len(self.unclean_char_list)}\n {sorted(self.unclean_char_list)}")
        print(f"\nCharacter list after cleaning: Size = {len(self.clean_char_list)}\n {sorted(self.clean_char_list)}")
        print(f"Sentences with acronyms elminated: {n}/{len(data)} ({round(n / len(data), 2) * 100}%)")
        return clean_data

def main():
    json = "example_eu.json"
    data = cu.read_manifest(f"./manifests/{json}")
    eu_normalizer = TextNormalizer(lang='eu', cp=False)
    clean_data = eu_normalizer.clean_sentences(data)
    json_clean=json.replace(".json","_clean.json")
    cu.write_manifest(f"./manifests/processed/{json_clean}", clean_data)

if __name__=="__main__":
    main()
