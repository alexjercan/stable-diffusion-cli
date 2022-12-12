# Stable Diffusion CLI

Tool that can be used in cli to generate images using stable diffusion.

Features:

1. Start just with the prompt and generate an image
2. Use an existing image and modify it using a prompt
3. Use the webcam to capture a frame and then apply the image to image pipeline

The tool will display the image after it is generated as a plot and it will save the png in the data directory, by default "data".

## Usage

Install requirements

```console
pip install -r requirements.txt
```

### Prompt to image

```console
python main.py --prompt "example of prompt"
```

### Image to image

```console
python main.py --prompt "example of prompt" --source image.png
```

### Webcam

```console
python main.py --prompt "example of prompt" --camera
```

Press `q` to take a frame and apply the pipeline to it.
