# SPDX-License-Identifier: Apache-2.0
"""Browse weather maps and sounding windows drawn by the engine."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

ROOT = Path(__file__).resolve().parent
CATALOG = json.loads((ROOT / "catalog.json").read_text())
SAMPLES = {row["id"]: row for row in CATALOG["samples"]}


def asset(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT / "assets") or not path.is_file():
        raise ValueError("The catalog names a missing sample asset")
    return str(path)


def fields(sample):
    return list(dict.fromkeys(row["variable"] for row in SAMPLES[sample]["maps"]))


def leads(sample, variable):
    return [row["lead"] for row in SAMPLES[sample]["maps"] if row["variable"] == variable]


def picture(sample, variable, lead):
    row = next(row for row in SAMPLES[sample]["maps"]
               if (row["variable"], row["lead"]) == (variable, lead))
    return asset(row["image"]), (f'**{SAMPLES[sample]["label"]}** · {variable} · {lead} · Valid {row["valid"]}'
                                f'\n\n{SAMPLES[sample].get("note", "")}')


def sample_changed(sample):
    variable = fields(sample)[0]
    lead = leads(sample, variable)[0]
    image, caption = picture(sample, variable, lead)
    return gr.Radio(choices=fields(sample), value=variable), gr.Radio(choices=leads(sample, variable), value=lead), image, caption


def variable_changed(sample, variable):
    lead = leads(sample, variable)[0]
    image, caption = picture(sample, variable, lead)
    return gr.Radio(choices=leads(sample, variable), value=lead), image, caption


SOUNDINGS = CATALOG["soundings"]
POINTS = list(dict.fromkeys(row["point"] for row in SOUNDINGS))


def sounding_leads(point):
    return [row["lead"] for row in SOUNDINGS if row["point"] == point]


def sounding(point, lead):
    row = next(row for row in SOUNDINGS if (row["point"], row["lead"]) == (point, lead))
    return asset(row["image"]), f'**{row["sample_label"]}** · {point} · {lead} · Valid {row["valid"]}', asset(row["json"])


def point_changed(point):
    lead = sounding_leads(point)[0]
    return gr.Radio(choices=sounding_leads(point), value=lead), *sounding(point, lead)


def build_app():
    first = next(iter(SAMPLES))
    variable = fields(first)[0]
    lead = leads(first, variable)[0]
    initial_image, initial_caption = picture(first, variable, lead)
    initial_sound, sound_caption, sound_json = sounding(POINTS[0], sounding_leads(POINTS[0])[0])
    with gr.Blocks(title="WOOF weather samples", theme=gr.themes.Soft(), analytics_enabled=False,
                   css=".gradio-container {max-width: 1400px !important} footer {display:none !important}") as demo:
        gr.Markdown("# WOOF weather samples\nArWen regional and hex global forecasts. Explore the maps and a full sounding window.")
        with gr.Tab("Maps"):
            sample = gr.Radio([(row["label"], row["id"]) for row in SAMPLES.values()], value=first, label="Sample")
            with gr.Row():
                field = gr.Radio(fields(first), value=variable, label="Variable")
                time = gr.Radio(leads(first, variable), value=lead, label="Lead time")
            caption = gr.Markdown(initial_caption)
            display = gr.Image(initial_image, label="Forecast map", type="filepath", interactive=False,
                               show_label=False, height=660, show_download_button=True,
                               show_fullscreen_button=True, show_share_button=False)
            gr.Markdown("Maps are drawn by the engine's Rust renderer. Each map includes a colour bar and units.")
            sample.input(sample_changed, sample, [field, time, display, caption])
            field.input(variable_changed, [sample, field], [time, display, caption])
            time.input(picture, [sample, field, time], [display, caption])
        with gr.Tab("Soundings"):
            gr.Markdown("Select a location to see a WOOF-ArWen sounding.")
            with gr.Row():
                point = gr.Radio(POINTS, value=POINTS[0], label="Point")
                sound_time = gr.Radio(sounding_leads(POINTS[0]), value=sounding_leads(POINTS[0])[0], label="Sounding lead time")
            sound_label = gr.Markdown(sound_caption)
            sound_display = gr.Image(initial_sound, interactive=False, show_label=False, height=720,
                                     show_download_button=True, show_fullscreen_button=True, show_share_button=False)
            profile = gr.File(sound_json, label="Profile and diagnostics (JSON)", interactive=False)
            gr.Markdown("Created with `woof soundings`. Open fullscreen to read the diagnostic tables.")
            point.input(point_changed, point, [sound_time, sound_display, sound_label, profile])
            sound_time.input(sounding, [point, sound_time], [sound_display, sound_label, profile])
        with gr.Tab("Run your own"):
            gr.Markdown(CATALOG["run_markdown"])
        gr.Markdown("Apache-2.0")
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    if port == 8080:
        raise ValueError("Choose a local port other than 8080")
    build_app().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"), server_port=port, share=False,
                       allowed_paths=[str(ROOT / "assets")])
