# 04 Checkpoint security

Checkpoints contain exactly `manifest.json` and `weights.pt`. The loader rejects
symlinks, hardlinks, ancestor symlinks, extra files, hash/size mismatches,
non-finite or abnormal tensors, dynamic models and unsafe pickle payloads. One
open descriptor is used for weights identity, hashing and weights-only loading.
