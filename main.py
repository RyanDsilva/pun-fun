import pprint
from setup import setup
import gradio as gr
from target_identification.target_identification import run_inference_pipeline
from sense_similarity.sense_similarity import (
    find_sense_similarity,
    find_source_pos,
    find_synset,
)
from sound_similarity_v2.sound_similarity_v2 import get_sound_similarity
from sound_similarity.sound_similarity import get_sound_similarity as get_sound_similarity_v1
from source_identification.inference import identify_source
from find_related_pun.classes import *
from find_related_pun import inference

from fastapi import FastAPI, Request
import uvicorn
import json

server = FastAPI()

pp = pprint.PrettyPrinter(indent=2)


def run(word, index = 0):
    result, _ = inference.find_top_n_puns(word, 5)
    pp.pprint(result)
    sentence = result[index]
    source_word = identify_source(str(sentence))
    if source_word is None or source_word == '':
        return (
            None,
            None,
            None,
            None,
            None,
            None,
        )
    res = run_inference_pipeline(str(sentence), source_word)
    for r in res:
        src = r["source"]
        pred = r["predicted"]
        sent = r["sentence"]
        pos = find_source_pos(src, sent)
        src_sense = find_synset(src)
        pred_sense = find_synset(pred)
        sense_sim = None
        if src_sense is not None and pred_sense is not None:
            print("Definitions")
            print(f"{src}: {src_sense.definition()}")
            print(f"{pred}: {pred_sense.definition()}")
            print("=" * 50)
            sense_sim = find_sense_similarity(src_sense, pred_sense)
            print(f"Sense similarity between {src} and {pred} is {sense_sim}")
        sound_sim, _, __ = get_sound_similarity(src, pred)
        sound_sim_v1 = get_sound_similarity_v1(src, pred)
        if sound_sim is not None:
            print(f"Sound similarity distance between {src} and {pred} is {sound_sim}")
        if sound_sim_v1 is not None:
            print(f"Sound similarity (ED) between {src} and {pred} is {sound_sim_v1}")
        print("=" * 50)
        if sense_sim is not None and sound_sim is not None:
            return (
                str(sentence),
                source_word,
                src_sense.definition(),
                pred,
                pred_sense.definition(),
                sense_sim,
                sound_sim,
            )
        elif sense_sim is None and sound_sim is not None:
            return (
                str(sentence),
                source_word,
                "No definition available",
                pred,
                "No definition available",
                "NA",
                sound_sim,
            )
        elif sense_sim is not None and sound_sim is None:
            return (
                str(sentence),
                source_word,
                src_sense.definition(),
                pred,
                pred_sense.definition(),
                sense_sim,
                "NA",
            )
        else:
            return (
                str(sentence),
                source_word,
                "No definition available",
                pred,
                "No definition available",
                "NA",
                "NA",
            )

@server.post("/get_pun")
async def getPun(data : Request):
    req_body = await data.json()
    pun, src, src_def, tgt, tgt_def, sense_sim, sound_sim = run(req_body['word'], req_body['index'])
    return {
        "status" : "SUCCESS",
        "result" : {
            "pun": pun,
            "src": src,
            "src_def": src_def,
            "tgt": tgt,
            "tgt_def": tgt_def,
            "sense": sense_sim,
            "sound": sound_sim,
        }
    }

if __name__ == "__main__":
    setup()
    app = gr.Interface(
        title="CNIT 519 - Project 3",
        fn=run,
        allow_flagging="never",
        inputs=[gr.Textbox(label="Input Word"), gr.Number(value=0, label='Top N-th result => N', precision=0)],
        outputs=[
            gr.Textbox(label="Pun"),
            gr.Textbox(label="Source"),
            gr.Textbox(label="Source Definition"),
            gr.Textbox(label="Target"),
            gr.Textbox(label="Target Definition"),
            gr.Textbox(label="Sense Similarity"),
            gr.Textbox(label="Sound Similarity Distance"),
        ],
    )
    # app.launch()
    uvicorn.run(server, port=3333)
    # print(run('politics', 1))
