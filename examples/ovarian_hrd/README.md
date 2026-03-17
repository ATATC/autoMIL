# Ovarian HRD Example

HRD status prediction from ovarian cancer WSIs. Binary classification
using CLAM-MB with H-optimus-1 encoder features.

This example shows only the `automil/` subdirectory that autoMIL adds to
an existing project. The full project would also contain:

- Model code (e.g., CLAM, nnMIL architectures)
- Data loading scripts
- Feature extraction pipeline
- Training scripts

## Results

- 189 experiments executed autonomously
- Best composite: 0.851 (from 0.814 baseline, +4.5%)
- Key discoveries: R-Drop, focal loss, gradient clipping, coordinate PE
- See `automil/graph.json` for the full experiment tree
