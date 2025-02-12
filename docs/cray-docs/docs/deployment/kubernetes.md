# Kubernetes Deployment

Craylm can be deployed on Kubernetes. This guide will walk you through the steps to deploy Cray on Kubernetes.

## Prerequisites

Make sure you have a kubernetes cluster running. You can set one up yourself or use a
managed service like GKE or AKS.

You can also use the provided Ansible recipe
to install Kubernetes on a Lambda VM running Ubuntu 22.04. The recipe is available at
[Ansible Kubernetes](https://github.com/cray-lm/cray-lm/deployment/ansible/k8.yml).

If you check out the repository on the server, you can run the playbook with the
following command:

```bash
ansible-playbook -i hosts -v k8.yml
```

The ansible playbook is a list of tasks that will install Kubernetes on the Lambda VM. You can inspect the playbook to see what it does and modify it to suit your needs.

It is also possible to run the playbook on a remote server by setting up the SSH connection in the `hosts` file.

```bash
[all]
lambda ansible_host=...
```

## Deploy Craylm

To deploy Craylm on Kubernetes, you need to clone the Cray repository and run the following command:

```bash
git clone git@github.com:cray-lm/cray-lm.git
cd cray-lm/deployment/helm/lambda
```

Edit the `values.yaml` file:

- set the `externalIP` address of the Lambda VM, or configure your ingress controller
- set the `model` to the huggingface model you want to deploy

Then run the following command to deploy Craylm:

```bash
helm install cray cray
```

## Verify the Deployment

To verify the deployment, you can run the following command:

```bash
kubectl get pods
```

You should see the Craylm pod running.

You can get the logs of the pod by running:

```bash
kubectl logs cray-...
```

