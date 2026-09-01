"""Generate one public deterministic StructField-Bench sample."""

from structfield.benchmarks import StructFieldBenchConfig, generate_sample

sample = generate_sample(
    "crossing",
    split="validation",
    index=0,
    config=StructFieldBenchConfig(height=32, width=32),
)
print(sample.metadata.sample_id, sample.image.shape, int(sample.valid_mask.sum()))
