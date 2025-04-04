import reflex as rx
from pathlib import Path
from FashionSearch.domain.services.process_service import ProcessService

class ProcessState(rx.State):
    """State for handling image processing."""
    image: str = ""  
    text: str = ""
    model: str = "Token 4"  # Default model
    results: list[dict] = []
    query_submitted: bool = False
    is_submittable: bool = False
    _temp_file: rx.UploadFile = None  # Store the file temporarily

    def check_submittable(self) -> None:
        """Check if both image and text are present."""
        self.is_submittable = bool(self._temp_file and self.text)

    @rx.event
    async def set_image(self, files: list[rx.UploadFile]):
        """
        Store the file immediately and show preview.
        Only read and write once here.
        """
        if not files or not isinstance(files, list) or len(files) == 0:
            self._temp_file = None
            self.image = ""
            return

        file = files[0]
        self._temp_file = file

        # Save to Reflex's upload directory
        upload_data = await file.read()
        filename = getattr(file, 'filename', None) or getattr(file, 'name', 'uploaded_file')
        outfile = rx.get_upload_dir() / filename

        # Write it to disk
        with outfile.open("wb") as file_object:
            file_object.write(upload_data)

        # Store just the filename
        self.image = filename
        self.check_submittable()

    @rx.event
    def set_text(self, text: str):
        """Update the text requirement."""
        self.text = text
        self.check_submittable()

    @rx.event
    def set_model(self, model: str):
        """Update the selected model."""
        self.model = model

    @rx.event
    async def submit(self):
        """Process the image and text."""
        print("Submit function called - checking requirements")

        if not self.is_submittable or not self._temp_file:
            print(f"Not submittable: is_submittable={self.is_submittable}, temp_file exists={self._temp_file is not None}")
            return 

        try:
            print('Starting submission process')
            # Get the full path for processing
            full_path = str((rx.get_upload_dir() / self.image).resolve())
            print(f"Full image path is: {full_path}")

            try:
                print('Calling ProcessService.process')
                self.results = ProcessService.process(full_path, self.text, self.model)
                print('Processing completed successfully!')
                self.query_submitted = True

            except ImportError as e:
                print(f"Import error: {e}")
                self.results = [{"image_url": "/error.png", "score": 0.0}]
                self.query_submitted = True

            except Exception as e:
                print(f"Error in processing: {e}")
                self.results = [{"image_url": "/error.png", "score": 0.0}]
                self.query_submitted = True
                raise

        except Exception as e:
            print(f"Error in submit: {e}")
            self.results = [{"image_url": "/error.png", "score": 0.0}]
            self.query_submitted = True
            raise

    @rx.event
    def clear_fields(self):
        """Reset all fields."""
        self.image = ""
        self._temp_file = None
        self.text = ""
        self.model = "Token 4"
        self.results = []
        self.query_submitted = False
        self.is_submittable = False

        from FashionSearch.ui.pages.home_page import UploadState
        UploadState.preview_url = ""
        UploadState.uploading = False
        UploadState.progress = 0

        return [
            rx.set_value("text_area", ""),
            rx.clear_selected_files("upload1"),
        ]
