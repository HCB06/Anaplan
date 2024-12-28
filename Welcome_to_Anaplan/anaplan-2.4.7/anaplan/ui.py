try:
    from moviepy.editor import VideoFileClip
    import sounddevice as sd
    import soundfile as sf
    import pkg_resources
except:
    pass

def loading_bars():
    
    GREEN = "\033[92m"
    RESET = "\033[0m"

    bar_format = f"{GREEN}{{bar}}{GREEN} {RESET} {{l_bar}} {{remaining}} {{postfix}}"
    bar_format_learner = f"{GREEN}{{bar}}{GREEN} {RESET} {{remaining}} {{postfix}}"

    return bar_format, bar_format_learner

def speak(message):

    message = pkg_resources.resource_filename('anaplan', f'{message}')
    video = VideoFileClip(message + ".mp4")

    audio = video.audio
    audio.write_audiofile("extracted_audio.wav")

    data, samplerate = sf.read("extracted_audio.wav")
    sd.play(data, samplerate)
    sd.wait()
    
