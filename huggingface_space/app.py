import os
import sys

os.environ["APP_ENV"] = "development"
os.environ["VIDEO_STORAGE_BACKEND"] = "s3"
os.environ.setdefault("VISUAL_MODEL_DIR", "/tmp/cricketcoach-model")
os.environ["CRICKETCOACH_DEVICE"] = "cuda"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import gradio as gr
import spaces

from app.ml.visual_classifier import get_visual_classifier


@spaces.GPU(duration=45)
def predict_video(video_path):
    if not video_path:
        raise gr.Error("Upload a supported cricket video.")
    classifier = get_visual_classifier()
    if not classifier.is_available:
        raise gr.Error("The private shot model could not be loaded from storage.")
    prediction = classifier.predict_video(video_path)
    if (
        not prediction.get("model_available")
        or not prediction.get("shot_label")
        or prediction.get("error")
    ):
        raise gr.Error("Shot classification failed. Please try another clip.")
    return prediction


with gr.Blocks(title="CricketCoach AI Inference") as demo:
    gr.Markdown(
        "# CricketCoach AI shot inference\n"
        "Private inference endpoint for the connected web application."
    )
    video = gr.Video(
        label="Video for shot classification",
        sources=["upload"],
        type="filepath",
    )
    result = gr.JSON(label="Prediction")
    classify = gr.Button("Classify shot", variant="primary")
    classify.click(predict_video, inputs=video, outputs=result, api_name="predict")

demo.queue(default_concurrency_limit=1).launch()
