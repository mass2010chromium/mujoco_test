# Jupyter notebook on SLURM via port forward

## Installation
Run the installer: `bash pace/install_scripts/install_notebook.sh`

(Assumes you have run all the other install steps. TODO: document)

## Running jupyter notebook server
1. Run (on login node):`pace/pace_python [-G <gpu>] --tunnel-port 8888 -m jupyter notebook`
    This does two things:
    - Spin up jupyter notebook
    - Open an SSH tunnel from the login node to the compute node

2. (on your local computer), run `ssh -vvv -NT -L 8888:127.0.0.1:8888 <user>@<login_node>.pace.gatech.edu
    - The login node should be the specific node that you are logged into.
      It will show on the terminal prompt: <user>@<login_node>
    - This ssh command on your local computer can stay running the whole time.
      It can be launched before or after the jupyter notebook

3. From the output of step 1, find the jupyter notebook link. Copy and paste it into a browser on your local computer.

That's all!
