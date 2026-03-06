"""Entry point for the Gradio application.

The app creates three tabs, each of which is implemented in its
own module so that the code remains clean and maintainable.
"""

import gradio as gr

from single_gen import create_tab as single_tab
from batch_gen import create_tab as batch_tab
from phasepaint_gen import create_tab as phasepaint_tab
from config import GALLERY_SIZE

css = f"""
#my_gallery {{
    width: {GALLERY_SIZE}px !important;
}}
"""

def main():
    with gr.Blocks(css=css) as demo:
        # create shared prompt/negative-prompt components that live
        # outside individual tabs; typing in one tab will update the
        # others automatically since they refer to the same component
        prompt_txt = gr.Textbox(label="Prompt", placeholder="Description of scene")
        neg_txt = gr.Textbox(label="Negative prompt", placeholder="Things to avoid")

        with gr.Tabs():
            with gr.TabItem("single gen"):
                single_tab(prompt_txt, neg_txt)
            with gr.TabItem("batch gen"):
                batch_tab(prompt_txt, neg_txt)
            with gr.TabItem("PhasePaint gen"):
                phasepaint_tab(prompt_txt, neg_txt)

    demo.launch()


if __name__ == "__main__":
    main()
