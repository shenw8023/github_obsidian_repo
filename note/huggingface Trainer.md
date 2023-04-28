
## 可重写
- **create_optimizer** — Sets up the optimizer if it wasn’t passed at init.
-   **create_scheduler** — Sets up the learning rate scheduler if it wasn’t passed at init.
-   **compute_loss** - Computes the loss on a batch of training inputs.
-   **training_step** — Performs a training step.
-   **prediction_step** — Performs an evaluation/test step.
-   **evaluate** — Runs an evaluation loop and returns metrics.
-   **predict** — Returns predictions (with metrics if labels are available) on a test set.


- Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.


## Trainer
- Parameters
	- model
	- args
	- ...
- 重要属性
	- model
	- model_wrapped
	- is_model_parallel
	- place_model_on_device
	- is_in_train
	- 

## TrainingArguments
TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop itself**.

Using [HfArgumentParser](https://huggingface.co/docs/transformers/v4.28.1/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the command line.