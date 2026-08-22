# Execution

> **Largely superseded by P1 (D-043).** The model table below is retired: see
> D-050 for the five evaluated checkpoints. The sampling, serving and cluster
> procedure is retired for the primary readout, which is a fixed padded forward
> pass under Transformers with no decoding, key-value cache or continuous
> scheduling (D-045). Manifest recording, revision pinning and the
> one-accelerator-per-model constraint still apply.

## Models

Two contrasting open-weight instruction-tuned families, recorded by exact model
identifier and access date. Family names without version identifiers are not
sufficient for reporting.

| Key | Model | Parallelism | Accelerators |
|---|---|---|---|
| `qwen3-32b` | `Qwen/Qwen3-32B` | single device | 1 × 80GB |
| `gemma3-27b` | `google/gemma-3-27b-it` | single device | 1 × 80GB |

Both models fit one accelerator in bfloat16 with room for an 8k key-value cache.
The pilot uses the 80GB pool because neither a two-accelerator 48GB request nor
a co-located 80GB pair was available. Single-device execution removes the
pairing constraint without changing the scientific configuration.

Qwen3 emits a reasoning block by default. It is disabled: the study scores
structured fields only, and an unconstrained preamble would break schema-
constrained decoding.

A 70B model is listed as deferred in `configs/models.yaml`. Adding it is a
confirmatory-run decision, taken only after the pilot shows the scenario set
avoids floor and ceiling effects on both families above.

## Sampling

Temperature 0.7, top-p 0.95, 300 maximum tokens, seeded per trial from a
recorded base.

Non-zero temperature is required for the run-to-run diagnostic to mean anything;
at temperature zero it measures nothing by construction and the free-parameter
control has to carry the determinacy check alone. If the pilot returns a
run-to-run standard deviation above ten points, the temperature is lowered and
the pilot re-run, because at that level the stochastic floor consumes the paired
difference the sample-size calculation depends on.

## Batching

Every prompt in the plan is submitted as one batch and the engine schedules it.
Two properties keep the cost down.

*Prefix reuse.* Repeated runs of a variant are byte-identical, and every second-
turn request contains its own first turn verbatim. With prefix caching enabled
the shared portion is prefilled once rather than once per request, which removes
most of the prefill work in the second pass.

*Constrained decoding.* Each response is decoded under its schema, so no
generation budget is spent on retries.

The two turns are sequential because the second must follow the first in the
same conversation. Within each turn, everything is submitted at once.

At pilot scale — twenty-four families, five conditions, two prestige levels, two
runs, plus twins — this is 528 generations per model per turn, or 1,056 across
both turns. Generation is minutes; weight download is tens of minutes. The
cache volume is what actually determines how long a run takes.

## Cluster jobs

Manifests are in `k8s/`. Every job runs the repository from the cache volume, so
the working tree is staged there once per run label before anything is submitted;
`k8s/pod-fetch.yaml` mounts the same claim for that copy and for recovering
outputs afterwards. Apply in order:

    kubectl apply -f k8s/pvc.yaml

    sed -e 's|__MODEL_ID__|Qwen/Qwen3-32B|' \
        -e 's|__MODEL_REVISION__|9216db5781bf21249d130ec9da846c4624c16137|' \
        k8s/job-fetch-weights.yaml | kubectl apply -f -

    sed -e 's|__MODEL_KEY__|qwen3-32b|' \
        -e 's|__MODEL_ID__|Qwen/Qwen3-32B|' \
        -e 's|__MODEL_REVISION__|9216db5781bf21249d130ec9da846c4624c16137|' \
        -e 's|__RUN_LABEL__|pilot01|' \
        k8s/job-run.yaml | kubectl apply -f -

    sed -e 's|__RUN_LABEL__|pilot01|' \
        k8s/job-analyse.yaml | kubectl apply -f -

A scenario set whose soft profiles have changed needs its twins authored again
before the plan will build, because the stored pool is refused at load time when
it is not the perturbation of the set it is loaded with:

    sed -e 's|__RUN_LABEL__|pilot01|' \
        k8s/job-twins.yaml | kubectl apply -f -

The exploratory criterion-reweighting contrast is a separate job so that it reads
completed responses without rewriting the analysis outputs that carry the
recorded gate verdict:

    sed -e 's|__RUN_LABEL__|pilot01|' \
        k8s/job-weighting.yaml | kubectl apply -f -

Notes that are easy to get wrong:

- **Use a Job, not an interactive pod.** Interactive pods are capped at two
  accelerators, 32GB of memory, and six hours, and are destroyed on that
  schedule regardless of what they are doing.
- **Requests equal limits.** Cluster policy requires limits within twenty
  percent of requests, and namespaces whose reservations consistently exceed
  their usage get throttled. The numbers in the manifest are what the run needs.
- **Weight download runs without an accelerator.** It is network- and
  disk-bound; requesting a GPU there holds one idle for the length of the
  download.
- **Inference reads the checkpoint from the cache volume in place.** Staging it
  to node-local disk first requires an ephemeral-storage reservation larger than
  most accelerator nodes advertise, so the job queues instead of running, and it
  moves the same sixty-five gigabytes twice.
- **Shared memory must be enlarged.** Tensor parallelism communicates between
  worker processes through `/dev/shm`, and the container default of 64MB is not
  enough. The failure presents as a hang during model load, not as an error.
- **Request only what the container writes.** Node-local storage now holds logs
  and an editable install rather than a checkpoint, so the reservation is small.
  Every accelerator node that cannot satisfy a request is a node the scheduler
  silently removes from consideration, and an over-large request is
  indistinguishable from a saturated cluster while the pod sits Pending.
- **A queued job is not a slow job.** `scripts/await_job.py` separates the two:
  a pod that has not been assigned a node within its scheduling guard is reported
  with the scheduler's reason and a non-zero exit rather than waited on. Use it
  in place of `kubectl wait`, whose completion condition cannot tell a pod that
  is working from one no node can accept.
- **The accelerator product and resource key are substituted at submission.** The
  scientific requirement is one accelerator of at least 80GB in bfloat16 on a
  single device; which product satisfies it is recorded in the run manifest, not
  fixed in advance. When the preferred pool is saturated, resubmitting against
  the wider set is a scheduling decision rather than a design change.
- **The plan is built on CPU before the model loads.** A scenario or integrity
  failure then stops the job in seconds rather than after the weights are
  resident.
- **Run the combined analysis after every model has finished.** The analysis
  job reads every response manifest under the label, writes the descriptive
  pilot outputs, and records the blocking diagnostic exit code alongside them.

The namespace is shared. Run one model at a time unless there is a reason not
to, and delete completed jobs rather than leaving them to expire.

## Recovering results

Outputs are written to the cache volume under `responses/<label>` and
`analysis/<label>`. Copy them out with `kubectl cp` from a short-lived pod that
mounts the same claim, then commit the analysis outputs. Raw responses are large
and are not committed; they stay on the volume, and the manifest records enough
to regenerate them.

## Run manifests

Every run writes `manifest__<model>__<label>.json` recording the exact model
identifier and checkpoint revision, inference-engine version, timestamps,
sampling settings, parallelism, dtype, context length,
whether structured decoding and prefix caching were active, the study seed, the
scenario set version, which name pool was used and whether it was provisional,
the accelerator product and memory class, and the host. A run without a manifest
is not a result.
