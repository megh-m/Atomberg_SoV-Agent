# Atomberg SoV Agent
## Setup 
Update setup.py with your own API Keys, especially the hugging face API Key as those get invalidated when made public.

Run setup.py by ``` $ python setup.py ``` in the terminal

This should create a ```.env``` file to securely store your API Keys while running
## Running
To run the agent in terminal use the command ``` $ python SoV_Agent_terminal.py ``` 

This. if it runs without error, will create two files, sov_chart.png and sov_results.csv that can be used for analysis 

Example files are attached in the repo

<b> If you are more comfortable running the scripts in Colab (for security concerns etc) use the ``` SoV_Agent_colab.ipynb``` notebook </b>

## Troubleshooting
### Module Import Errors 
Ensure that the setup.py runs correctly. You might need to update certain packages, like torch
### API Key errors 
It is recommended you create your own API keys for <b>Cohere</b>, <b>Youtube</b> and <b>HuggingFace</b>. Creating one for Hugging Face is mandatory. 

## Further Scope
While this implementation sources only from Youtube, other Networking platforms can be integrated by adding their respective API keys and implementing pipleines to extract from them. 

The SoV analysis is done in a combined fashion, taking in all inputs

