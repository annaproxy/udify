from collections import defaultdict
no_train_set = ["UD_Breton-KEB", "UD_Faroese-OFT", "UD_Swedish-PUD"]
no_train_set_lowercase = [ "br_keb-ud","fo_oft-ud","sv_pud-ud"]

languages = [
    "UD_Arabic-PADT",
    "UD_Armenian-ArmTDP",
    "UD_Breton-KEB",
    "UD_Bulgarian-BTB",
    "UD_Buryat-BDT",
    "UD_Czech-PDT",
    "UD_English-EWT",
    "UD_Faroese-OFT",
    "UD_Finnish-TDT",
    "UD_French-Spoken",
    "UD_German-GSD",
    "UD_Hindi-HDTB",
    "UD_Hungarian-Szeged",
    "UD_Italian-ISDT",
    "UD_Japanese-GSD",
    "UD_Kazakh-KTB",
    "UD_Korean-Kaist",
    "UD_Norwegian-Nynorsk",
    "UD_Persian-Seraji",
    "UD_Russian-SynTagRus",
    "UD_Swedish-PUD",
    "UD_Tamil-TTB",
    "UD_Telugu-MTG",
    "UD_Upper_Sorbian-UFAL",
    "UD_Urdu-UDTB",
    'UD_Vietnamese-VTB'
]

languages_readable = [
    "Arabic",
    "Armenian",
    "Breton",
    "Bulgarian",
    "Buryat",
    "Czech",
    "English",
    "Faroese",
    "Finnish",
    "French",
    "German",
    "Hindi",
    "Hungarian",
    "Italian",
    "Japanese",
    "Kazakh",
    "Korean",
    "Norwegian",
    "Persian",
    "Russian",
    "Swedish",
    "Tamil",
    "Telugu",
    "UpperSorbian",
    "Urdu",
    'Vietnamese'
]

languages_lowercase = [
    "ar_padt-ud",
    "hy_armtdp-ud",
    "br_keb-ud",
    "bg_btb-ud",
    "bxr_bdt-ud",
    "cs_pdt-ud",
    "en_ewt-ud",
    "fo_oft-ud",
    "fi_tdt-ud",
    "fr_spoken-ud",
    "de_gsd-ud",
    "hi_hdtb-ud",
    "hu_szeged-ud",
    "it_isdt-ud",
    "ja_gsd-ud",
    "kk_ktb-ud",
    "ko_kaist-ud",
    "no_nynorsk-ud",
    "fa_seraji-ud",
    "ru_syntagrus-ud",
    "sv_pud-ud",
    "ta_ttb-ud",
    "te_mtg-ud",
    "hsb_ufal-ud",
    "ur_udtb-ud",
    "vi_vtb-ud",
]

final_languages = languages[-6:]
final_languages_lowercase = languages_lowercase[-6:]

limits = defaultdict(lambda:5)
limits2 = {
    "UD_Breton-KEB":2,
    "UD_Buryat-BDT":1,
    "UD_Faroese-OFT":3,
    "UD_Kazakh-KTB":1,
    "UD_Swedish-PUD":3,
    "UD_Upper_Sorbian-UFAL":1
}
limits.update(limits2)
