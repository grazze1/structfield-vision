# 11 Supply chain

uv 0.12.3 produces Python 3.11 CPU and CUDA 12.6 PEP 751 locks with complete
hashes and PyTorch 2.13.0 profile builds. CycloneDX 1.6 and license inventories
must reproduce exactly; `UNKNOWN`, audit findings, secrets, private paths,
restricted assets, large files or wheel leaks block release.

Final qualification result: **PASS**. Re-resolution with uv 0.12.3 produced
identical dependency bodies for both locks. Both CycloneDX 1.6 SBOMs and both
license inventories reproduced exactly with no `UNKNOWN` entry. pip-audit
2.10.1 reported no known vulnerabilities in either locked profile. Repository
and wheel scans found no private path, credential pattern, restricted asset,
or oversized tracked file.
