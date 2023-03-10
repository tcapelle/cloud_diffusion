# Cloud Diffusion Experiment

This codebase contains an implementation of a deep diffusion model applied to cloud images. It was developed as part of a research project exploring the potential of diffusion models
for image generation and forecasting.

## Setup

1. Clone this repository
2. Install the required libraries: `pip install -e .` or `pip install cloud_diffusion`
3. Set up your WandB account by signing up at [wandb.ai](https://wandb.ai/site).
4. Set up your WandB API key by running `wandb login` and following the prompts.

## Usage

To train the model, run `python train.py`. You can play with the parameters on top of the file to change the model architecture, training parameters, etc.

You can also override the configuration parameters by passing them as command-line arguments, e.g.
`python train.py --epochs=10 --batch_size=32`.

## License

This code is released under the [MIT License](LICENSE).