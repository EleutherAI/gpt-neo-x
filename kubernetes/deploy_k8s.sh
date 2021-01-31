#!/usr/bin/env bash

#  --- USAGE ---
# $ deploy_k8.sh [branch=main] [n_nodes=4] [name_suffix=$USER] [image]
# You need to install yq

# Check yq
yq &> /dev/null || { echo 'You need to install `yq >= v4`. `brew install yq` or `pip install yq`' ; exit 1; }

BRANCH=${1:-main}
N_NODES=${2:-4}
SUFFIX=${3:-$(whoami)}
IMAGE=$4

DEPLOYMENT_NM='neox-'"$SUFFIX"
WD=`dirname "$BASH_SOURCE"`

echo BRANCH $BRANCH. N-NODES $N_NODES. DEPLOYMENT NAME $DEPLOYMENT_NM.
if [ -n "$IMAGE" ]
  then
    echo "DOCKER IMAGE $IMAGE."
fi

# Generate ssh key pair and post start script
echo Generate SSH key pair
ssh-keygen -t rsa -f $WD/id_rsa -N ""

post_start_script="
cp /secrets/id_rsa.pub /root/.ssh/authorized_keys;
chmod 600 /root/.ssh/authorized_keys;
chmod 700 /root/.ssh;
chown -R root /root/.ssh;
rm -r /app/*;
cd /app;
git clone --single-branch --branch $BRANCH https://github.com/EleutherAI/gpt-neox.git .;
"
echo $post_start_script > $WD/post_start_script.sh

# Add ssh key to k8 secrets and post start script
DATE=$(date +%s)
SECRET_NM="$DEPLOYMENT_NM-$DATE"
kubectl create secret generic $SECRET_NM \
  --from-file=id_rsa.pub=$WD/id_rsa.pub \
  --from-file=post_start_script.sh=$WD/post_start_script.sh

# Template k8 configuration
yq e '.metadata.name = "'"$DEPLOYMENT_NM"\" $WD/k8s_spec.yml |
yq e '.spec.replicas = '"$N_NODES" - |
yq e '.spec.template.spec.volumes[1].secret.secretName = "'"$SECRET_NM"\" - > $WD/k8s_spec_temp.yml

if [ -n "$IMAGE" ]
  then
    yq e -i '.spec.template.spec.containers[0].image = "'"$IMAGE"\" $WD/k8s_spec_temp.yml
fi

exit

# Delete previous and setup deployment
kubectl delete deploy/$DEPLOYMENT_NM 2&> /dev/null || { echo 'No previous deployment'; }
kubectl apply -f $WD/k8s_spec_temp.yml

echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/$DEPLOYMENT_NM || { echo 'Deployment failed' ; exit 1; }

echo Generate hosts file
kubectl get pods -o wide | grep $DEPLOYMENT_NM | awk '{print $6 " slots=8"}' > $WD/hostfile
export MAIN_ID=$(kubectl get pods | grep $DEPLOYMENT_NM | awk '{print $1}' | head -n 1)

echo Copying ssh key and host file to main node:
echo $MAIN_ID
kubectl cp $WD/hostfile $MAIN_ID:/job
kubectl cp $WD/id_rsa $MAIN_ID:/root/.ssh

rm $WD/id_rsa* $WD/hostfile $WD/k8s_spec_temp.yml $WD/post_start_script.sh

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash
