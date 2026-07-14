# Security Policy: Dexplore

NVIDIA is dedicated to the security and trust of our software products and services,
including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points
outlined below. **Please do not report security vulnerabilities through GitHub
issues or pull requests.**

## Reporting a Vulnerability

To report a potential security vulnerability in Dexplore:

* **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
* **E-Mail:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to use the following PGP key for secure email communication:
    [NVIDIA public PGP Key](https://www.nvidia.com/en-us/security/pgp-key)
* **GitHub:** Use this repository's **Security** tab > **Report a vulnerability** to
  submit a report directly.

Please include the following information:

- Product/project name and version or branch affected
- Type of vulnerability (e.g., code execution, denial of service, data tampering)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept code (if available)
- Potential impact assessment

**Detailed reports help NVIDIA evaluate and address issues faster.**

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge receipt
of your report, validate the vulnerability and assess severity, develop and test a
fix, and publish a security bulletin as appropriate.

## Security Architecture & Context

Dexplore is a research codebase (CoRL 2025) for training and evaluating
dexterous-manipulation policies in NVIDIA Isaac Gym. It implements a two-stage
pipeline: a state-based teacher policy trained with reinforcement learning
(`dexplore/run.py`, built on `rl_games` PPO), and a vision-based student policy
distilled with DAgger using a PointNet++ encoder (`dexplore/run.py --distill`,
`dexplore/eval_distill.py`). A separate data-processing pipeline
(`data_processing/convert_grab.py`) converts the third-party GRAB motion-capture
dataset into training data.

This software operates at the **CLI tool / research library** level. It runs
locally on a single-user workstation with an NVIDIA GPU. It exposes no network
services, has no authentication or authorization layer, stores no credentials,
and processes no data other than files the user supplies. Its primary security
responsibility is the handling of user-supplied artifacts — serialized model
checkpoints, motion datasets, robot-description and mesh assets, and YAML
configuration — all of which influence or execute code in the user's process
once loaded.

**Repository Exposure Classification:** Public.
Basis: open-source release under the NVlabs organization on github.com; document
written to public-safe detail.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: externally distributed open-source research software that ships pretrained
model checkpoints; no live service is operated, and all data is processed locally
by the user.

Data enters the system through: CLI arguments and YAML configuration files
(parsed with `yaml.SafeLoader`); serialized artifacts loaded with pickle-based
deserializers (`.pth`/`.pt` checkpoints and interaction data via `torch.load`,
`.npz` dataset archives via `numpy` with `allow_pickle=True`); simulation assets
(URDF, OBJ, STL, GLB) parsed by Isaac Gym's native importers and `trimesh`; and
one HTTPS download of a SMPL-X vertex-segmentation JSON during data processing.
All of these inputs are fully trusted once the user supplies them — there is no
validation, signing, or sandboxing layer.

### Threat Model

The following scenarios represent the primary security concerns for this project:

1. **Arbitrary code execution via malicious model checkpoints:** Checkpoints are
   loaded with pickle-based deserialization (`torch.load` and `rl_games`
   `torch_ext.load_checkpoint`) in `dexplore/learning/dexplore_agent.py`,
   `dexplore/learning/dexplore_players.py`, `dexplore/eval_distill.py`, and
   `dexplore/env/tasks/dexplore_distill.py`. Loading a `.pth` checkpoint obtained
   from an untrusted source can execute arbitrary code with the invoking user's
   privileges. Only load the checkpoints shipped with this repository or ones you
   trained yourself.

2. **Arbitrary code execution via untrusted dataset files:** The data pipeline
   loads `.npz` archives with `np.load(..., allow_pickle=True)`
   (`data_processing/convert_grab.py`) and preprocessed interaction data with
   `torch.load` (`dexplore/env/tasks/base_dexplore_task.py`). A tampered GRAB
   sequence, object archive, or interaction-data file can execute arbitrary code
   during conversion or environment creation. Obtain GRAB and SMPL-X data only
   from their official distribution sites.

3. **Malformed simulation assets exploiting native parsers:** Robot URDFs and
   object meshes (OBJ/STL/GLB/PLY) are parsed by Isaac Gym's native asset
   importer and by `trimesh`. Crafted asset files from untrusted sources could
   trigger parser defects in native code, causing crashes (denial of service) or
   potentially memory corruption.

4. **Tampered runtime download in the data pipeline:**
   `data_processing/smpl_constants.py` downloads a SMPL-X vertex-segmentation
   JSON from a third-party GitHub repository over HTTPS without checksum or
   signature verification. A compromised upstream file would silently corrupt
   data-processing output (integrity impact; the file is parsed as JSON, not
   executed).

5. **Vulnerabilities in pinned, end-of-life dependencies:** `requirement.txt`
   pins older package versions (e.g., `numpy==1.21.1`, `protobuf==3.20.0`,
   `rl-games==1.1.4`) and the supported runtime is Python 3.8 with Isaac Gym
   Preview 4, both of which no longer receive security updates. Known CVEs in
   these pinned versions will not be patched here; users should isolate the
   environment (e.g., a dedicated conda env or container) rather than share it
   with other workloads.

### Critical Security Assumptions

- **All input artifacts are trusted.** Checkpoints, dataset files, interaction
  data, simulation assets, and YAML configs are deserialized or parsed without
  validation, signing, or sandboxing. The software assumes the user only supplies
  files from trusted origins (this repository's release artifacts, official
  dataset distributions, or the user's own training runs).
- **Trusted, single-user execution environment.** The software runs with the
  invoking user's privileges on a workstation the user controls. There is no
  authentication, authorization, or multi-tenant isolation, and none is intended.
- **No network exposure.** The software opens no listening sockets. Auxiliary
  tools the user may launch alongside it (e.g., TensorBoard on training logs)
  bind at the user's discretion and are outside this project's security boundary.
- **Trusted dependency installation.** Dependencies are assumed to be installed
  from trusted sources (PyPI, and Isaac Gym from NVIDIA's developer program) with
  no integrity verification beyond the package manager's defaults.
- **Third-party dataset licensing and handling.** GRAB and SMPL-X are licensed
  third-party datasets containing human motion-capture data. Users obtain them
  directly from their publishers and are responsible for complying with the
  applicable licenses and handling requirements; this repository redistributes
  neither.

## Scope

Dexplore is a research artifact released for reproducibility and further
research. It is provided as-is under the terms of the [LICENSE](LICENSE) file and
is not intended for production deployment or for processing untrusted input.
