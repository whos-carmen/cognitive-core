#!/bin/bash
# Mount S3 bucket for training data and checkpoints.
# Run on instance start (or add to crontab).
set -euo pipefail

MOUNT_POINT="/mnt/s3"
BUCKET="cognitive-core-checkpoints"

if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "S3 already mounted at $MOUNT_POINT"
    exit 0
fi

# Get credentials from instance metadata
CREDS=$(curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/cognitive-core-s3)
if [ -z "$CREDS" ]; then
    echo "ERROR: No IAM credentials available"
    exit 1
fi

ACCESS_KEY=$(echo $CREDS | python3 -c 'import sys,json; print(json.load(sys.stdin)["AccessKeyId"])')
SECRET_KEY=$(echo $CREDS | python3 -c 'import sys,json; print(json.load(sys.stdin)["SecretAccessKey"])')
TOKEN=$(echo $CREDS | python3 -c 'import sys,json; print(json.load(sys.stdin)["Token"])')

echo "${ACCESS_KEY}:${SECRET_KEY}:${TOKEN}" | sudo tee /etc/passwd-s3fs > /dev/null
sudo chmod 600 /etc/passwd-s3fs

sudo mkdir -p "$MOUNT_POINT"
sudo s3fs "$BUCKET" "$MOUNT_POINT"     -o passwd_file=/etc/passwd-s3fs     -o url=https://s3.us-east-1.amazonaws.com     -o use_path_request_style     -o allow_other     -o umask=000     -o iam_role=auto     -o ensure_diskfree=10000

echo "S3 mounted at $MOUNT_POINT"
ls "$MOUNT_POINT/"
