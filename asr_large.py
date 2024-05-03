from transformers import pipeline
import gradio as gr

def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(filepath)
    return output["text"]

if __name__ == "__main__":
    asr = pipeline(task="automatic-speech-recognition",
               model="./models/openai/whisper-large-v3",
               device=0)
    
    demo = gr.Blocks()
    mic_transcribe = gr.Interface(
        fn=transcribe_speech,
        inputs=gr.Audio(sources="microphone",
                        type="filepath"),
        outputs=gr.Textbox(label="Transcription",
                        lines=3),
        allow_flagging="never")
    file_transcribe = gr.Interface(
        fn=transcribe_speech,
        inputs=gr.Audio(sources="upload",
                        type="filepath"),
        outputs=gr.Textbox(label="Transcription",
                        lines=3),
        allow_flagging="never",
    )
    with demo:
        gr.TabbedInterface(
            [mic_transcribe,
            file_transcribe],
            ["Transcribe Microphone",
            "Transcribe Audio File"],
        )

    demo.launch(server_name="0.0.0.0",
                share=True, 
                server_port=8888)
    
    demo.close()
