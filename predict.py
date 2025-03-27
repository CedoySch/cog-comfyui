# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
            self.comfyUI.handle_weights(
                workflow,
                weights_to_download=[],
            )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
            self,
            input_file: Path,
            filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Обновление фото человека (узел 17)
        if kwargs.get("person_image_filename"):
            workflow["17"]["inputs"]["image"] = kwargs["person_image_filename"]

        # Обновление фото одежды (узел 54)
        if kwargs.get("cloth_image_filename"):
            workflow["54"]["inputs"]["image"] = kwargs["cloth_image_filename"]

        # Обновление типа одежды (узел 369)
        if kwargs.get("cloth_type"):
            workflow["369"]["inputs"]["cloth_type"] = kwargs["cloth_type"]

        # Обновление seed (узел 263) если он используется
        if kwargs.get("seed"):
            workflow["263"]["inputs"]["seed"] = kwargs["seed"]

    def predict(
            self,
            person_image: Path = Input(
                description="Фотография человека для примерки одежды",
                default=None,
            ),
            cloth_image: Path = Input(
                description="Изображение одежды для наложения",
                default=None,
            ),
            cloth_type: str = Input(
                description="Тип одежды (upper для верхней одежды, lower для нижней)",
                default="upper",
                choices=["upper", "lower", "overall"]
            ),
            output_format: str = optimise_images.predict_output_format(),
            output_quality: int = optimise_images.predict_output_quality(),
            seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        # Обработка фото человека
        person_image_filename = None
        if person_image:
            person_image_filename = self.filename_with_extension(person_image, "person_image")
            self.handle_input_file(person_image, person_image_filename)

        # Обработка фото одежды
        cloth_image_filename = None
        if cloth_image:
            cloth_image_filename = self.filename_with_extension(cloth_image, "cloth_image")
            self.handle_input_file(cloth_image, cloth_image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            person_image_filename=person_image_filename,
            cloth_image_filename=cloth_image_filename,
            cloth_type=cloth_type,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )