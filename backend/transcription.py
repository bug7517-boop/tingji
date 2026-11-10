"""Detect language per speech region so a language switch cannot hide later speech."""
import re


def text_language(text):
    chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    english = bool(re.search(r'[A-Za-z]', text))
    if chinese and english:
        return 'mixed'
    return 'zh' if chinese else 'en' if english else 'other'


def transcribe_regions(model, pcm, language, cancelled=lambda: False):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    regions = get_speech_timestamps(pcm, VadOptions(
        min_silence_duration_ms=500, speech_pad_ms=300, max_speech_duration_s=30))
    for region in regions:
        if cancelled():
            raise InterruptedError()
        # zh is the bilingual UI mode: do not force Chinese on English speech.
        detect = language in {'zh', 'auto'}
        selected_language = 'zh' if language == 'zh-only' else language
        segments, _ = model.transcribe(pcm[region['start']:region['end']],
            language=None if detect else selected_language, multilingual=detect,
            beam_size=5, vad_filter=False, condition_on_previous_text=False, task='transcribe')
        offset = region['start'] / 16000
        for segment in segments:
            if cancelled():
                raise InterruptedError()
            yield offset + segment.start, offset + segment.end, segment.text
