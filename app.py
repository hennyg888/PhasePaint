"""Entry point for the Gradio application.

The app creates three tabs, each of which is implemented in its
own module so that the code remains clean and maintainable.
"""

import gradio as gr

from single_gen import create_tab as single_tab
from batch_gen import create_tab as batch_tab
from phasepaint_gen import create_tab as phasepaint_tab

def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("single gen"):
                single_tab()
            with gr.TabItem("batch gen"):
                batch_tab()
            with gr.TabItem("PhasePaint gen"):
                phasepaint_tab()

    demo.launch()


if __name__ == "__main__":
    main()
