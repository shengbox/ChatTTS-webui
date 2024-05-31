import ChatTTS
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import datetime
import sys
import os
import asyncio
import torch
import logging

logger = logging.getLogger('     ')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Text2Speech(BaseModel):
    text: str
    voice: int
    prompt: str


model_path = os.path.join(os.path.dirname(__file__), 'models')

chat = ChatTTS.Chat()
chat.load_models(
    vocos_config_path=os.path.join(model_path, 'config/vocos.yaml'),
    vocos_ckpt_path=os.path.join(model_path, 'asset/Vocos.pt'),
    dvae_config_path=os.path.join(model_path, 'config/dvae.yaml'),
    dvae_ckpt_path=os.path.join(model_path, 'asset/DVAE.pt'),
    gpt_config_path=os.path.join(model_path, 'config/gpt.yaml'),
    gpt_ckpt_path=os.path.join(model_path, 'asset/GPT.pt'),
    decoder_config_path=os.path.join(model_path, 'config/decoder.yaml'),
    decoder_ckpt_path=os.path.join(model_path, 'asset/Decoder.pt'),
    tokenizer_path=os.path.join(model_path, 'asset/tokenizer.pt'),
)


@app.post("/generate")
async def generate_text(request: Text2Speech):
    text = request.text
    torch.manual_seed(request.voice)
    std, mean = torch.load(os.path.join(model_path, 'asset/spk_stat.pt')).chunk(2)
    rand_spk = torch.randn(768) * std + mean
    params_infer_code = {
        'spk_emb': rand_spk,
        'temperature': 0.1,
        'top_P': 0.7,
        'top_K': 20,
    }

    params_refine_text = {
        # 'prompt': '[oral_2][laugh_0][break_6]'
        'prompt': request.prompt
    }

    wavs = await asyncio.to_thread(chat.infer, text, use_decoder=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
    audio_data = np.array(wavs[0])
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=0)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    output_file = f'outputs/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.wav'
    sf.write(output_file, audio_data.T, 24000)
    return FileResponse(output_file, media_type='audio/wav', filename='generated_audio.wav')

if __name__ == "__main__":
    logger.info('node-server running on http://0.0.0.0:3000 \n')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
