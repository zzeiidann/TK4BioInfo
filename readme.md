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

Output:
```
Alignment Score: 2.0
Type: global
Identity: 44.44%
Similarity: 66.67%
Gaps: 4
Range: [0-8] x [0-9]

=========
Position 1-12
Seq1: -ACGTGG-TT--
      | |.  | ||  
Seq2: GC-TTTTGTA--
=========
```

### Local Alignment

```python
# Perform local alignment (Smith-Waterman)
result = pairwise("ACGTGGTT", "GCTTTTGTA", mode="local")
result.view()
```

### Custom Scoring Parameters

```python
from TK4Bioinfo.seq_alignment import PairwiseAligner

# Create aligner with custom parameters
aligner = PairwiseAligner(
    match_score=5.0,
    mismatch_score=-4.0,
    gap_open=-5.0,
    gap_extend=-2.0
)

# Align sequences
result = aligner.align("ACGTACGT", "ACGTCCGT", mode="global")
print(result)
```

## Advanced Usage

### Batch Alignment with Threading

```python
from TK4Bioinfo.seq_alignment import PairwiseAligner

# Define multiple sequence pairs
sequence_pairs = [
    ("ACGTACGT", "ACGTACGT"),
    ("AAAAAAA", "TTTTTTT"),
    ("GCTAGCT", "GCTCGCT"),
]

# Perform batch alignment using 4 worker threads
aligner = PairwiseAligner()
results = aligner.align_batch(sequence_pairs, mode="global", n_workers=4)

for i, result in enumerate(results):
    print(f"Alignment {i+1}: Identity={result.identity:.2%}")
```

### Asynchronous Alignment

```python
import asyncio
from TK4Bioinfo.seq_alignment import PairwiseAligner

async def align_sequences():
    aligner = PairwiseAligner()
    
    # Align multiple pairs concurrently
    sequence_pairs = [
        ("ACGTACGT", "ACGTCCGT"),
        ("GGGGGGGG", "CCCCCCCC"),
    ]
    
    results = await aligner.align_batch_async(sequence_pairs, mode="global")
    return results

# Run async alignment
results = asyncio.run(align_sequences())
```

### Accessing Alignment Statistics

```python
result = pairwise("ACGTACGT", "ACGTCCGT", mode="global")

# Access detailed statistics
print(f"Alignment Score: {result.score}")
print(f"Identity: {result.identity:.2%}")
print(f"Similarity: {result.similarity:.2%}")
print(f"Number of Gaps: {result.gaps}")
print(f"Aligned Sequence 1: {result.seq1_aligned}")
print(f"Match String: {result.match_string}")
print(f"Aligned Sequence 2: {result.seq2_aligned}")
```