import whisper

# load model
model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
# discusses mel: https://learnopencv.com/automatic-speech-recognition/

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
# whisper.detect_language(), whisper.decode() provide lower-level access to the model.

# tokenize while transcribe?
def get_tokenizer(model, language):
    multilingual = not model.endswith(".en")
    return whisper.tokenizer.get_tokenizer(multilingual=False, language="french", task="transcribe")

# print the recognized text
print(result.text)


print(model)