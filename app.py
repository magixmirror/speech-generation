import os
import glob
from datetime import datetime
import numpy as np

import torch
import torchaudio

torchaudio.set_audio_backend("soundfile")
import nltk

from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import SUPPORTED_LANGS

import scipy.io.wavfile as wav

from denoiser import pretrained
from denoiser.dsp import convert_audio

import gradio as gr

nltk.download("punkt")

preload_models()


def generate_and_save_audio(text, selected_speaker, text_temp, waveform_temp, apply_denoise):
    sentences = nltk.sent_tokenize(text)

    chunks = [""]
    token_counter = 0

    for sentence in sentences:
        current_tokens = len(nltk.Text(sentence))
        if token_counter + current_tokens <= 250:
            token_counter = token_counter + current_tokens
            chunks[-1] = chunks[-1] + " " + sentence
        else:
            chunks.append(sentence)
            token_counter = current_tokens

    # Generate audio for each prompt
    audio_arrays = []
    for prompt in chunks:
        audio_array = generate_audio(prompt, selected_speaker, text_temp, waveform_temp)
        audio_arrays.append(audio_array)

    # Combine the audio files
    combined_audio = np.concatenate(audio_arrays)

    # save the audio to a file
    static_folder = os.path.join(os.getcwd(), "static")
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"output_{timestamp}.wav"
    filepath = os.path.join(static_folder, filename)
    wav.write(filepath, SAMPLE_RATE, combined_audio)

    model = pretrained.dns64().cuda()
    wavFile, sr = torchaudio.load(filepath)
    wavFile = convert_audio(wavFile.cuda(), sr, SAMPLE_RATE, model.chin)
    with torch.no_grad():
        denoised = model(wavFile[None])[0]
        denoised_filename = f"denoised_output_{timestamp}.wav"
        denoised_filepath = os.path.join(static_folder, denoised_filename)
        torchaudio.save(denoised_filepath, denoised.cpu(), sr)

    denoised_filepath = None
    if apply_denoise:
        model = pretrained.dns64().cuda()
        wavFile, sr = torchaudio.load(filepath)
        wavFile = convert_audio(wavFile.cuda(), sr, SAMPLE_RATE, model.chin)
        with torch.no_grad():
            denoised = model(wavFile[None])[0]
            denoised_filename = f"denoised_output_{timestamp}.wav"
            denoised_filepath = os.path.join(static_folder, denoised_filename)
            torchaudio.save(denoised_filepath, denoised.cpu(), sr)

    return f"./static/{filename}", f"./static/{denoised_filename}" if denoised_filepath else None


speakers_list_v1 = []
speakers_list_v2 = []

for lang, code in SUPPORTED_LANGS:
    for n in range(10):
        speakers_list_v2.append(f"v2/{code}_speaker_{n}")
        speakers_list_v1.append(f"{code}_speaker_{n}")

custom_prompts = []
# Define the directory
dir_path = 'custom_prompts'

# Use glob to get all the .npz files in the directory
custom_prompts = glob.glob(os.path.join(dir_path, '*.npz'))
custom_prompts.sort()

speakers_list = custom_prompts + speakers_list_v2 + speakers_list_v1

input_text = gr.Textbox(label="Input Text", lines=4, placeholder="Enter text here...")
text_temp = gr.Slider(
    0.1,
    1.0,
    value=0.7,
    label="Generation Temperature",
    info="1.0 more diverse, 0.1 more conservative",
)
waveform_temp = gr.Slider(
    0.1,
    1.0,
    value=0.7,
    label="Waveform temperature",
    info="1.0 more diverse, 0.1 more conservative",
)
apply_denoise = gr.Checkbox(label="Apply Denoiser", default=False)
output_audio = gr.Audio(label="Generated Audio", type="filepath")
clean_audio = gr.Audio(label="Denoised Audio", type="filepath")
speaker = gr.Dropdown(speakers_list, value=speakers_list[0], label="Acoustic Prompt")

io = gr.Interface(
    generate_and_save_audio,
    inputs=[input_text, speaker, text_temp, waveform_temp, apply_denoise],
    outputs=[output_audio, clean_audio],
    title="Generate Audio",
    description="Enter text and hear the generated audio.",
    theme="default",
    examples=[["Greetings Earthling, I am Jim, an extraterrestrial being from the planet Zog in the Andromeda galaxy."],["""Once upon a time, there was a curious little fox named Freddie who loved to explore. One day, he stumbled upon a beautiful meadow and decided to take a nap.

As he slept, a group of fireflies fluttered above him and sang a sweet lullaby:

"Close your eyes and go to sleep,
Let your worries softly keep,
In the morning, when you wake,
A brand new day will soon await."

Freddie felt warm and safe, and soon he was sound asleep. From that day on, whenever he felt tired or scared, he would think back to the fireflies and their beautiful lullaby.

And so, my dear child, whenever you need a good night's rest, just remember Freddie and his firefly friends. Close your eyes and let their sweet lullaby carry you off to a peaceful sleep."""],
["""Once upon a time, in a faraway land, there lived a little bunny named Benny. Benny was a curious bunny who loved to explore the forest and all its wonders. He would spend hours hopping around, smelling the flowers, and playing with his animal friends.

One night, as Benny was getting ready for bed, he noticed a bright star shining through his window. He gazed at it for a while, feeling peaceful and content. Suddenly, he had an idea.

He decided to make a wish upon the star. "I wish I could explore the moon," he whispered.

The next morning, Benny woke up early and hopped outside. To his amazement, he saw a spaceship waiting for him. Without hesitation, Benny hopped in and took off into the sky.

As he soared towards the moon, Benny marveled at the stars and planets around him. He felt weightless and free, and he couldn't help but giggle with delight.

When he finally landed on the moon, Benny couldn't believe his eyes. The moon was made of soft, fluffy cheese, and there were little mice scurrying around everywhere! Benny played with the mice and ate some cheese, feeling happy and fulfilled.

Eventually, it was time to go home. Benny hopped back into the spaceship and flew back to Earth. As he drifted off to sleep that night, he felt grateful for the amazing adventure he had just experienced.

And so, Benny lived happily ever after, always dreaming of his next space adventure. The end."""]],
)

io.launch()
