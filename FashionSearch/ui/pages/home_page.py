import reflex as rx
from FashionSearch.ui.states.process_state import ProcessState
from typing import Callable
from pathlib import Path

import os

class UploadState(rx.State):
    """State for handling client-side preview."""
    preview_url: str = ""
    # You can remove uploading/progress if not needed

    @rx.event
    def set_preview_url(self, url: str):
        """Set the preview URL immediately after file upload."""
        self.preview_url = url

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s)."""
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # Save the file
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

        # Any additional logic here (e.g. storing for preview)...

def build_result(result: dict):
    """Build the result component."""
    return rx.vstack(
        rx.image(
            src=result["image_url"],  # Use the URL directly
            width="200px",
            height="auto",
            object_fit="contain",
        ),
        rx.text(
            f"Score: {result['score']:.2f}", 
            text_align="center"
        ),
        spacing="4",
        align_items="center",
    )

def header() -> rx.Component:
    """Header component."""
    return rx.box(
        rx.heading(
            "Multi-model Fashion Search",
            font_size="2em",
            padding="0.5em",
            text_align="center",
            color="white",
        ),
        width="100%",
        bg="rgb(107,99,246)",
    )

def upload_section() -> rx.Component:
    """Upload section component."""
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "Upload Image",
                    color="rgb(107,99,246)",
                    bg="white",
                    border="1px solid rgb(107,99,246)",
                    width="100%",  # Make button full width
                ),
                rx.cond(
                    ProcessState.image,
                    rx.image(
                        src=rx.get_upload_url(ProcessState.image),
                        width="100%",
                        height="auto",
                        object_fit="contain",
                    ),
                ),
            ),
            id="upload1",
            accept={
                "image/png": [".png"],
                "image/jpeg": [".jpg", ".jpeg"],
            },
            max_files=1,
            on_drop=ProcessState.set_image(
                rx.upload_files(upload_id="upload1")
            ),
            border="1px dotted rgb(107,99,246)",
            padding="1em",
            width="100%",  # Consistent width for upload component
        ),
        width="100%",
        spacing="4",
    )

def input_section() -> rx.Component:
    """Input section component."""
    return rx.vstack(
        rx.text_area(
            id="text_area",
            placeholder="Describe your requirement",
            on_blur=ProcessState.set_text,
            width="100%",
            min_height="100px",
            margin_top="1em",
            border_radius="0.5em",
        ),
        rx.select(
            ["Token 4", "Token 1"],
            value=ProcessState.model,
            on_change=ProcessState.set_model,
            width="100%",
            margin_top="1em",
            border="1px solid rgb(107,99,246)",
            border_radius="0.5em",
        ),
        rx.hstack(
            rx.button(
                "Submit",
                on_click=ProcessState.submit,
                variant="solid",
                color_scheme="purple",
                is_disabled=~ProcessState.is_submittable,
                width="50%",  # Half width
            ),
            rx.button(
                "Reset",
                on_click=ProcessState.clear_fields,
                variant="solid",
                color_scheme="red",
                width="50%",  # Half width
            ),
            spacing="4",
            margin_top="1em",
            width="100%",  # Full width container
        ),
        width="100%",
        spacing="4",
        align_items="center",
    )

def results_section() -> rx.Component:
    """Results section component."""
    return rx.box(
        rx.cond(
            ProcessState.query_submitted,
            rx.vstack(
                rx.text(
                    "Process is completed",
                    color="green",
                    font_size="1.2em",
                    margin_bottom="1em",
                ),
                rx.text(
                    "Results:", 
                    font_size="1.5em", 
                    font_weight="bold",
                    margin_bottom="1em",
                ),
                rx.hstack(
                    rx.foreach(
                        ProcessState.results,
                        build_result,
                    ),
                    spacing="4",
                    flex_wrap="wrap",
                    justify_content="center",
                ),
                width="100%",
                align_items="flex-start",
            ),
            rx.center(
                rx.text(
                    "Submit a query to see results",
                    color="gray.500",
                    font_style="italic",
                ),
            ),
        ),
        width="70%",
        height="100%",
        padding="2em",
    )

@rx.page("/")
def home_page() -> rx.Component:
    """The main page."""
    return rx.vstack(
        # Fixed header
        header(),
        
        # Scrollable content area
        rx.box(
            rx.hstack(
                # Left panel
                rx.box(
                    rx.vstack(
                        upload_section(),
                        input_section(),
                        width="100%",
                        spacing="4",
                        padding="2em",
                        align_items="center",
                    ),
                    width="30%",  # Keep consistent panel width
                    height="100%",
                    border_right="1px solid #eaeaea",
                ),
                # Right panel
                results_section(),
                width="100%",
                spacing="0",
            ),
            width="100%",
            height="calc(100vh - 4em)",
            overflow_y="auto",  # This makes only this container scrollable
        ),
        
        width="100%",
        height="100vh",
        spacing="0",
        overflow="hidden",  # Prevents page-level scrollbar
    )
