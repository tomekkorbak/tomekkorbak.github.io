---
layout: post
title: Tips on setting up a GPU cluster on Google Kubernetes Engine
share: false
tags: [Kubernetes, Google Kubernetes Engine, GPU, Google Kubernetes Engine, Google Cloud Platform, cluster autoscaling, node taints]

---

This blog post is a bunch of unstructured notes to my future self on setting up a virtual GPU cluster for machine learning research (i.e. running experiments) managed by [Kubernetes](https://kubernetes.io/). My experience is with [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine), but most of the tips below should generalise to other cloud providers.

## Why bother setting up a Kubernetes cluster for machine learning research?

The use case I had in mind is when you need to run a lot of machine learning experiments but don’t have access to a good physical GPU cluster. While a single, physical workstation is usually fine for prototyping and debugging, once you need more thorough evaluation (e.g. sweeps over multiple random seeds and hyperparameters or longer runs) setting up a cluster has numerous benefits:

1. Parallelisation. Instead of using one machine for 10 hours, you could use 10 machines for an hour and shorten your feedback cycle 10x. My experience is that sometimes the ability to run a large number of jobs in parallel is a game-changer in terms of productivity.
2. Scalability. Ultimately, it’s the allure of being able to upscale your computational resources almost without limit, with almost zero overhead.
3. Portability. In contrast with buying and setting up a physical machine, the overhead of setting up a virtual cluster is minimal. You don’t need to worry about keeping it running and are not dependent on support staff. Moreover, configuration and automation infrastructure around a virtual cluster is easy to move and share across teams, locations, cloud providers, organisations and funding sources.
4. Abstraction. You are not tied to particular hardware which will go out-of-date one day. Kubernetes seems pretty well-adopted these days so I’d predict a Kubernetes configuration file will go stale slower than physical a GPUs.
5. Cost-effectiveness. It depends on your circumstances, but cloud tends to be cheaper than physical hardware. It’s true overall but especially when you don’t pay for idle time and you can further optimise costs by using spot VMs.
6. Carbon footprint. You can (and should) set up your nodes in regions where [most of the energy in the grid is clean](https://cloud.google.com/sustainability/region-carbon). 
7. Job scheduling and monitoring. I think Kubernetes offers better user experience than for instance [slurm](https://en.wikipedia.org/wiki/Slurm_Workload_Manager) in terms of documentation, configurability and automation.

## What’s a good Docker base image with CUDA and MuJoCo?

I used this [one](https://github.com/jannerm/trajectory-transformer/blob/master/azure/Dockerfile). One change is that now you can use the [public MuJoCo key](https://roboti.us/file/mjkey.txt).

## How to automate job submission?

I use my fork of [mrunner](https://github.com/deepsense-ai/mrunner) (an unsupported experiment management tool) which — with Kubernetes backend — roughly does the following:

1. Creates a Dockerfile by filling a Jinja template with your base image and paths to your dependencies, source code files and command to run.
2. Builds a Docker image based on that Dockerfile. The entrypoint is the command running the script.
3. Pushes the image to Google Container Registry.
4. If not present, creates a project-specific namespace.
5. If not present, creates a pod which runs an [NFS server](https://github.com/kubernetes/examples/tree/master/staging/volumes/nfs) to share mounted volume available to future jobs.
6. Creates Kubernetes Volume and Container objects, and out of them, a PodSpec object which is then wrapped into a JobSpec object. Finally, a Job object is created and submitted.

The preconditions for all that is creating a cluster. I do that manually with `gcloud container clusters create`.

## How can jobs use GPUs on GPUs nodes?

First, you need to make sure there are GPU nodes in the node pool. On Google Cloud, you probably also need to set up [regional quotas](https://cloud.google.com/compute/quotas) for GPU nodes. Also, you can’t use the [Autopilot](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) cluster because it doesn’t support GPU nodes (at least at the time of writing that).

Then, you need to enable CUDA drivers on GPU nodes. This commands does the trick for me:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Finally, your job must request GPU as a resource.

## How to monitor and debug runs?

For general cluster utilisation monitoring, I use `kubectl get jobs` and sometimes `kubectl describe node`. For debugging, `kubectl logs` is nice. For monitoring the actual experiment, I just use [Weights and Biases](https://wandb.ai). 

## How to autoscale GPU nodes to 0

That’s tricky because for me GPU nodes live indefinitely even after all the jobs are done. That’s probably because some `kube-system` pods get allocated there. Unfortunately, failure to scale down defeats the main purpose of setting up a virtual cluster: not having to pay for idle GPU nodes.

My solution is to create a cluster with two node pools: a GPU node pool (with autoscaling and minimum number of nodes set to 0) and an admin pool, with a single, cheap CPU node which we can afford to run indefinitely. Then, you need to use [node taints and tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/) to mark GPU nodes with a keyword that only GPU jobs tolerate. I set up node taints for the GPU node pool when creating the cluster and have corresponding tolerations as part of my PodSpec. Even with GPU nodes tainted, you might still need to wait 10 minutes after a job finishes for downscaling the GPU node pool.

The [Autopilot](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) cluster could be a cleaner solution but it doesn’t support GPU nodes yet.