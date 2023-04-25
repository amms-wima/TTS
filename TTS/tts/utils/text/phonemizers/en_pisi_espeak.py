from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak
from TTS.tts.utils.text.phonemizers.pa_si_phonemizer import pali_to_ipa

import re

'''
Hoping that this is not a re-entrant class
'''
class En_with_PiSi_ESpeak(ESpeak):
    _pre_phonemized_text = False

    def _phonemize_preprocess(self, text):
        self._pre_phonemized_text = False
        ipa_sub = text[:6]
        # print(f"text: [{text}]")
        # print(f"startWith: {ipa_sub}")
        if text.startswith("ipa,//"):
            text = text.replace("ipa,//", "")
            self._pre_phonemized_text = True
            # print(f"_phonemize_preprocess> _pre_phonemized_text: {self._pre_phonemized_text}")
            return [text], []
        # print(f"_phonemize_preprocess> _pre_phonemized_text: {self._pre_phonemized_text}")
        return super()._phonemize_preprocess(text)

    def _phonemize(self, text, separator=None):
        # print(f"_phonemize> _pre_phonemized_text: {self._pre_phonemized_text}")
        if (self._pre_phonemized_text):
            return text
        
        ret = ''
        sections = re.split(r'[@]', text)
        pali_subsections = re.findall(r'@([^@]+)@', text)
        for i, subsection in enumerate(sections):
            if (subsection == ''):
                continue
            if (subsection in pali_subsections):
                ret += pali_to_ipa(subsection)
            else:
                trimmed_text = subsection.strip()
                if (trimmed_text in [',', '.']):
                    ret += subsection
                else:
                    en_phonemization = super()._phonemize(subsection, separator) 
                    ret += en_phonemization
        return ret

        # en_phonemization = super()._phonemize(text, separator) 
        # return en_phonemization
        # ret = ''
        # sections = re.split(r'[@]', text)
        # pali_subsections = re.findall(r'@([^@]+)@', text)
        # for i, subsection in enumerate(sections):
        #     if (subsection in pali_subsections):
        #         ret += pali_to_ipa(subsection)
        #     else:
        #         en_phonemization = super()._phonemize(subsection, separator) 
        #         ret += en_phonemization
        # return ret
    
    def _phonemize_postprocess(self, phonemized, punctuations) -> str:
        # print(f"_phonemize_postprocess> _pre_phonemized_text: {self._pre_phonemized_text}")
        ret = super()._phonemize_postprocess(phonemized, punctuations)
        # print(f"ipa://{ret}")
        return ret
    
    def phonemize(self, text: str, separator="|", language: str = None) -> str:
        ret = super().phonemize(text, separator, language)
        # print(f"before final encoding: ({ret})")
        return ret
