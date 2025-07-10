# Corpus Processing Utils
This repository gives you tools to manage multiple features when processing audio/text datasets mainly for creating STT and ASR corpus.
The data is managed using the Nvidia NeMo's "manifest.json" format.

**Supported features:**
- Languages:
  - Spanish 'ES'.
  - Basque 'EU'.
- Text format:
  - With Capitalization & Punctuation.
  - Normalized, lowercase text without punctuation characters.
- Audio format:
  - WAV.
  - MP3.

**Unsupported features:**
- Numeric character management:
  - Work in progress for expansion: 
    - 'eu': '123' --> 'Ehun eta hogeita hiru', 'es': '123' --> 'Ciento veintitrés'
    - 'eu': '3.' --> 'Hirugarrena', 'es': '3º' --> 'Tercero'
    - 'eu': '1996/05/23' --> 'Mila bederatziehun eta laurogeita hamaseiko maiatzaren hogeitahirua', 'es': '23/05/1996' --> 'Veintitrés de mayo de mil novecientos noventaiseis'
- Acronyms and abbreviations management:
  - Work in progress for expansion:
    - 'eu': 'EHU' --> 'E hatxe u / Euskal Herriko Unibertsitatea', 'es': 'UPV' --> 'U pe uve / Universidad del País Vasco'
    - 'eu': 'km' --> 'kilometroa', 'es': 'km' --> 'kilómetro'