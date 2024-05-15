#!/bin/sh

track_bucket_dir=$HOME/advanced
team_bucket_dir=$HOME/nyanplan3
nsc_bucket_dir=$HOME/nsc
gcsfuse_conf=/workspaces/til24-vlm/.devcontainer/hooks/gcsfuse.yaml

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

sudo apt-get update
sudo apt-get install -y gcsfuse

if [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
  echo "Authenticating GCloud, user input is required!"
  gcloud auth application-default login
fi

echo "Mounting GCSFuse!"

mkdir -p $track_bucket_dir
mkdir -p $team_bucket_dir
mkdir -p $nsc_bucket_dir

echo user_allow_other | sudo tee -a /etc/fuse.conf > /dev/null
echo til-ai-24-advanced $track_bucket_dir gcsfuse config_file=$gcsfuse_conf,ro,allow_other,implicit_dirs,uid=1000,gid=1000,_netdev | sudo tee -a /etc/fstab
echo nyanplan3-til-ai-24 $team_bucket_dir gcsfuse config_file=$gcsfuse_conf,rw,allow_other,implicit_dirs,uid=1000,gid=1000,_netdev | sudo tee -a /etc/fstab
echo til-ai-24-data $nsc_bucket_dir gcsfuse config_file=$gcsfuse_conf,ro,allow_other,implicit_dirs,uid=1000,gid=1000,_netdev | sudo tee -a /etc/fstab
mount -a

echo "Configuring Docker Artifact Registry!"

gcloud auth configure-docker asia-southeast1-docker.pkg.dev -q
gcloud config set artifacts/location asia-southeast1
gcloud config set artifacts/repository repository-$1
