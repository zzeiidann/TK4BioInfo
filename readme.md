# TK4Bioinfo

A high-performance bioinformatics toolkit for Python, providing efficient implementations of sequence alignment algorithms with support for concurrent processing.

 alignment statistics

## Installation

### From GitHub

```bash
git clone https://github.com/zzeiidann/TK4Bioinfo.git
cd TK4Bioinfo
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20.0

## Quick Start

### Basic Usage

```python
from TK4Bioinfo.seq_alignment import pairwise

# Perform global alignment
result = pairwise("ACGTGGTT", "GCTTTTGTA", mode="global")

# Display alignment with visualization
result.plot()
```
