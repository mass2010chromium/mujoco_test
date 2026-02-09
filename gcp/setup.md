1. Create GCP machine
  - Should have (for now) at least 30GB VRAM, min. chunk size 20
2. Set up ssh to GCP
  - ssh to the machine via GCP web interface
  - Record its IP
    - `curl -s ifconfig.co | tee ip_address`
  - In GCP web portal, copy in your ssh public key
  - Now, you should be able to connect: `ssh <your_computer_user>@<gcp_ip>`
    - In general your local computer user will not be the same as the GCP user.
  - For now, ssh, and use `sudo su <gcp_user>` to switch
3. Set up group
  - Create a new group for everyone who wants to use the machine (or just you).
    - I use name "shared"
    - `sudo addgroup shared`
  - Add both the GCP user and your local user to the group
    - `sudo adduser <username> shared`
    - Log out and log back in for group changes to take into effect. (Includes relaunching tmux windows)
  - Designate one user as the "host"
    - TODO: Can this be just a shared folder? I think there is still some danger (.cache/.ollama)
    - Change permissions for their home directory:
      - `sudo chgrp -R shared <home_directory_path>`
      - `sudo chmod -R g+s <home_directory_path>`
4. Link workspace folder
  - Assumes that you previously placed code in a folder in the "host"'s home directory (`~/workspace`)
  - In your user (in shared group, but not the host):
    - `ln -sT <path_to_host_home_directory>/workspace workspace`
