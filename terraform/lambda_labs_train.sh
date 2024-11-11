#!/bin/bash

# NOTE: Turn this back on or script debugging
# set -x

# The following command line arguments are supported:
#   lambda_api_key
#   lambda_key_name
#   lambda_private_key
#   config_path
#   num_training_steps

# Default Values
num_training_steps="-1"
nproc_per_node="1"

# Loop through the arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --lambda_api_key)
      lambda_api_key="$2"
      shift # past argument
      shift # past value
      ;;
    --lambda_key_name)
      lambda_key_name="$2"
      shift
      shift
      ;;
    --lambda_private_key)
      lambda_private_key="$2"
      shift
      shift
      ;;
    --config_path)
      config_path="$2"
      shift
      shift
      ;;
    --num_training_steps)
      num_training_steps="$2"
      shift
      shift
      ;;
    --nproc_per_node)
      nproc_per_node="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Check to make sure the arguments are not empty
check_empty() {
  local var_name="$1"
  local var_value="$2"

  if [ -z "$var_value" ]; then
    echo "Error: '$var_name' variable is empty or not set."
    exit 1
  fi
}

check_empty "lambda_api_key" "$lambda_api_key"
check_empty "lambda_key_name" "$lambda_key_name"
check_empty "lambda_private_key" "$lambda_private_key"
check_empty "config_path" "$config_path"

# First tar up the current repository, so that we have the source code
# we want to run
SOURCE_DIRS=(
    "configs"
    "evaluation"
    "minllm"
    "sampling"
    "training"
    "requirements_notorch.txt"
)
ARCHIVE_NAME=$(mktemp)

tar -czf "$ARCHIVE_NAME" "${SOURCE_DIRS[@]}"
echo "Source code archived to $ARCHIVE_NAME"

# List the available user instances
# Lambda Labs API endpoint to list instances
api_url="https://cloud.lambdalabs.com/api/v1/instance-types"

# Fetch instances list
response=$(curl -s -H "Authorization: Bearer $lambda_api_key" "$api_url")

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  echo "Error: jq is not installed. Please install jq to parse JSON."
  exit 1
fi

# Parse and display instance details (assuming instance has "id" and "status" fields)
echo "Available Instances:"
echo "$response" | jq -r '.data | to_entries[] | "\(.value.instance_type.name): \(.value.regions_with_capacity_available[].name // "No available regions") [\(.value.instance_type.price_cents_per_hour) cents per hour]"'

read -p "Which GPU type do you want?: " gpu_type
read -p "Which region do you want?: " lambda_region

# Create and launch the requested infrastructure on Lambda Labs
terraform -chdir="$(pwd)/terraform/lambdalabs" init
terraform -chdir="$(pwd)/terraform/lambdalabs" apply -var="LAMBDALABS_API_KEY=$lambda_api_key" -var="gpu_type=$gpu_type" -var="lambda_key_name=$lambda_key_name" -var="lambda_region=$lambda_region" -var="lambda_private_key=$lambda_private_key" -auto-approve

# Obtain the IP address of the running instance
INSTANCE_IP=$(terraform -chdir="$(pwd)/terraform/lambdalabs" output -raw instance_ip_addr)
echo "Terraform instance launched on $INSTANCE_IP"

# Now that the instance is loaded, send the source code over
scp -i $lambda_private_key -o StrictHostKeyChecking=no $ARCHIVE_NAME "ubuntu@${INSTANCE_IP}:/home/ubuntu/minllm.tar.gz"

REMOTE_SCRIPT="
#!/bin/bash
python -m venv minllm_env
source minllm_env/bin/activate
pip install --upgrade pip
mkdir minllm; mv minllm.tar.gz minllm;cd minllm;tar -xzvf minllm.tar.gz
pip install -r requirements_notorch.txt
pip install torch torchvision
pip install --upgrade nvidia-nccl-cu12
export PYTHONPATH=.
torchrun --standalone --nproc-per-node $nproc_per_node --nnodes=1 training/train.py --config_path $config_path --num_training_steps $num_training_steps
"
echo "$REMOTE_SCRIPT" | ssh -i $lambda_private_key -o StrictHostKeyChecking=no  ubuntu@$INSTANCE_IP 'bash -sx'

# TODO: Download the results from the remote machine

# Destroy the running resources at the end
terraform -chdir="$(pwd)/terraform/lambdalabs" destroy -var="LAMBDALABS_API_KEY=$lambda_api_key" -var="gpu_type=$gpu_type" -var="lambda_key_name=$lambda_key_name" -var="lambda_region=$lambda_region" -var="lambda_private_key=$lambda_private_key" -auto-approve
