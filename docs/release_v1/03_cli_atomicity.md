# 03 CLI atomicity and integrity

The POSIX publisher uses private staging, file and directory fsync, immutable
generation rename, serialized writers, and an atomic pointer switch. The
resolver validates ancestry, regular singly linked files, schema, declarations,
sizes and SHA-256. Unknown user files are preserved.
