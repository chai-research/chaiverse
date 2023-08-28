#!/bin/bash

set -e

TAG=$1

if [ -z "$TAG" ]; then
    echo "No tag provided" 2>&1
    exit 1
fi

IMAGE="chaiverse/runpod:$TAG"
echo "Building image '$IMAGE'"

docker build --no-cache -t "$IMAGE" .
docker push "$IMAGE"
