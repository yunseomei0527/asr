import time
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(preds, refs):
    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    return {"WER": wer, "CER": cer}


def compute_rtf(process_time, audio_duration):
    return process_time / audio_duration
