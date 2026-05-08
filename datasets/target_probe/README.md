# Pi0 "Where am I going" probe

Steps:
1. run `libero_get_targets.py <dataset_name>` to generate LLM annotations for the goal objects for each sequence.
   Requires OpenRouter API key.
   dataset\_name is one of the libero datasets, ex. `libero_90`.
   This will generate a JSON file in the `libero_targets` folder.
2. run `record_target_data.py <dataset_name>` to generate activations from the target network.
   This will take a long time -- it has to run and the simulation multiple times for each task in the dataset.
   This will generate bunch of array dump files in the `outputs_and_transforms` folder.
3. run `group_data.py <dataset_name>` to collect activations to train with.
   You can edit the script -- by default we take only last token activations.
   This will generate training input and output data in the toplevel folder.
4. run `train_probe.py <dataset_name>` to collect activations to train with.
   The architecture definition is in `probe_network.py`.
   This will (stupidly) generate a checkpoint in `checkpoints/state`.
5. run `test_probe.py <dataset_name>` to see results in the terminal...
   
