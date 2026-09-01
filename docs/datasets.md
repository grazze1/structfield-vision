# Dataset provenance and use boundaries

StructField-Vision 1.0.0 distributes no dataset, clinical spreadsheet,
organizer image or label, pretrained weight, or checkpoint. Dataset adapters
operate only on paths supplied by the user and perform strict validation at
that external boundary.

## Historical sources

Historical training and experiments used two externally governed sources:

1. Qiao, Cao, Hou, et al., Scientific Data (2026),
   [DOI 10.1038/s41597-026-07418-6](https://doi.org/10.1038/s41597-026-07418-6).
   The versioned [Zenodo record](https://zenodo.org/records/18779434), DOI
   `10.5281/zenodo.18779434`, states CC BY 4.0 for the data record. Users must
   preserve attribution and verify the record's current terms. The article's
   content license must not be inferred from the data-record license.
2. Zemborain et al., Cornea (2023),
   [DOI 10.1097/ICO.0000000000003319](https://doi.org/10.1097/ICO.0000000000003319).
   The official [SNP-Net page](https://people.duke.edu/~sf59/Zemborain_SNP_Net.htm)
   requires academic-research-only use and citation of the paper.

Apache-2.0 covers neither source. A user who supplies external data is
responsible for access authorization, citation, privacy, ethics, and usage
restrictions.

## Reproduction boundary

The active tests use synthetic fixtures and do not read either source. No
1.0.0 claim depends on silently downloading data. Historical result documents
are retained as records, not regenerated evidence.
