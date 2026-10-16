---
title: CricketCoach AI Shot Inference
emoji: 🏏
colorFrom: green
colorTo: gray
sdk: gradio
python_version: 3.10
app_file: app.py
---

Private ZeroGPU inference service for the CricketCoach AI web app.

Create this Space as **private** and select **ZeroGPU** hardware. Add the
Supabase S3 endpoint, region, bucket, access key ID, and secret access key as
Space secrets. The model weights remain in the private Supabase bucket; they
are not committed to this Space.
