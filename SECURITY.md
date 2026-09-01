# Security policy

Security fixes are supported for the latest 1.x release. Report suspected
vulnerabilities privately to the maintainers through the repository host's
private security-reporting mechanism when one is configured. Do not include
patient data, credentials, proprietary weights, or exploit payloads in a public
issue.

Checkpoint loading accepts only the documented directory schema and
`weights_only=True` tensor state dictionaries. Never treat a legacy pickle as a
safe StructField-Vision checkpoint. CLI bundles and checkpoints should remain on
trusted local filesystems; network filesystem atomicity is outside the v1 claim.
