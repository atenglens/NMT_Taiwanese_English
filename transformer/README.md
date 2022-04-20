- run_trainer.py training script for translation from Hugging Face transformers library:
https://github.com/huggingface/transformers

- Code added is denoted by the comment block:
###### @atenglens ######

########################

Added translation generation for validation dataset.
Added print statement to display # of trainable parameters.

- Dataset: https://huggingface.co/datasets/atenglens/taiwanese_english_translation

- results folder contains model config files and translation generations
- generated_predictions.txt contains predicted translations of test set
- predictions.txt contains predicted translations of validation set
- references.txt contains target translations of validation set
- target translations of test set can be found in the Hugging Face dataset above
