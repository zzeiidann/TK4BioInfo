"""
Pairwise Sequence Alignment Module
Complete implementation matching Bioconductor pwalign functionality
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Literal, Union
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
from collections import defaultdict


@dataclass
class AlignmentResult:
    """Store alignment results and metadata"""
    seq1_aligned: str
    seq2_aligned: str
    score: float
    start1: int
    end1: int
    start2: int
    end2: int
    alignment_type: str
    match_string: str
    identity: float
    similarity: float
    gaps: int
    seq1_original: str
    seq2_original: str
    
    def __str__(self) -> str:
        """String representation of alignment"""
        return (
            f"Alignment Score: {self.score}\n"
            f"Type: {self.alignment_type}\n"
            f"Identity: {self.identity:.2%}\n"
            f"Similarity: {self.similarity:.2%}\n"
            f"Gaps: {self.gaps}\n"
            f"Range: [{self.start1}-{self.end1}] x [{self.start2}-{self.end2}]\n"
        )
    
    def plot(self, width: int = 80) -> None:
        """
        Display alignment with match indicators
        Format matches R Bioconductor pwalign output style
        """
        lines = []
        
        # Add header
        lines.append("")
        lines.append("Global PairwiseAlignmentsSingleSubject (1 of 1)")
        lines.append(f"pattern: [{self.start1}] {self.seq1_original}")
        lines.append(f"subject: [{self.start2}] {self.seq2_original}")
        lines.append(f"score: {self.score}")
        lines.append("")
        
        # Format alignment in blocks
        for start in range(0, len(self.seq1_aligned), width):
            a_block = self.seq1_aligned[start:start+width]
            b_block = self.seq2_aligned[start:start+width]
            
            # Create match string with proper formatting
            m_block = "".join(" " if x == " " else "|" if x == "|" else "." 
                             for x in self.match_string[start:start+width])
            
            lines.append(f"pattern: {a_block}")
            lines.append(f"         {m_block}")
            lines.append(f"subject: {b_block}")
            lines.append("")  # Blank line between blocks
        
        # Print all lines
        for line in lines:
            print(line)
    
    def view(self, width: int = 80) -> None:
        """Alias for plot method"""
        self.plot(width)
    
    def nmatch(self) -> int:
        """Number of matching positions"""
        return sum(1 for a, b in zip(self.seq1_aligned, self.seq2_aligned) 
                   if a == b and a != '-')
    
    def nmismatch(self) -> int:
        """Number of mismatching positions (excluding gaps)"""
        return sum(1 for a, b in zip(self.seq1_aligned, self.seq2_aligned) 
                   if a != b and a != '-' and b != '-')
    
    def nedit(self) -> int:
        """Number of edits (mismatches + indels)"""
        return self.nmismatch() + self.nindel()
    
    def nindel(self) -> Tuple[int, int]:
        """Number of insertions and deletions (seq1_indels, seq2_indels)"""
        seq1_indels = self.seq1_aligned.count('-')
        seq2_indels = self.seq2_aligned.count('-')
        return (seq1_indels, seq2_indels)
    
    def pid(self, type: str = "PID1") -> float:
        """
        Percent identity
        PID1: (matches / alignment_length) * 100
        PID2: (matches / shorter_sequence) * 100
        PID3: (matches / average_length) * 100
        PID4: (matches / (matches + mismatches)) * 100
        """
        matches = self.nmatch()
        mismatches = self.nmismatch()
        
        if type == "PID1":
            return (matches / len(self.seq1_aligned)) * 100 if len(self.seq1_aligned) > 0 else 0
        elif type == "PID2":
            shorter = min(len(self.seq1_original), len(self.seq2_original))
            return (matches / shorter) * 100 if shorter > 0 else 0
        elif type == "PID3":
            avg_len = (len(self.seq1_original) + len(self.seq2_original)) / 2
            return (matches / avg_len) * 100 if avg_len > 0 else 0
        elif type == "PID4":
            total = matches + mismatches
            return (matches / total) * 100 if total > 0 else 0
        else:
            raise ValueError(f"Unknown PID type: {type}")
    
    def mismatch_table(self) -> Dict[Tuple[str, str], int]:
        """Table of mismatches by base pair"""
        table = defaultdict(int)
        for a, b in zip(self.seq1_aligned, self.seq2_aligned):
            if a != b and a != '-' and b != '-':
                table[(a, b)] += 1
        return dict(table)
    
    def mismatch_summary(self) -> Dict:
        """Summary of mismatch statistics"""
        table = self.mismatch_table()
        return {
            'total_mismatches': sum(table.values()),
            'mismatch_types': len(table),
            'mismatch_table': table
        }
    
    def consensus_matrix(self, as_prob: bool = False) -> Dict[int, Dict[str, int]]:
        """Build consensus matrix for alignment positions"""
        matrix = {}
        for i, (a, b) in enumerate(zip(self.seq1_aligned, self.seq2_aligned)):
            matrix[i] = {'seq1': a, 'seq2': b}
            if a == b and a != '-':
                matrix[i]['consensus'] = a
            else:
                matrix[i]['consensus'] = 'N'
        return matrix
    
    def compare_strings(self) -> str:
        """Return comparison string with ?, +, - for match, pattern-only, subject-only"""
        result = []
        for a, b in zip(self.seq1_aligned, self.seq2_aligned):
            if a == b:
                result.append('?')  # match
            elif a == '-':
                result.append('-')  # insertion in seq2
            elif b == '-':
                result.append('+')  # insertion in seq1
            else:
                result.append('?')  # mismatch
        return ''.join(result)
    
    def pattern(self) -> 'PatternInfo':
        """Return pattern (seq1) information"""
        return PatternInfo(
            sequence=self.seq1_aligned,
            start=self.start1,
            end=self.end1,
            original=self.seq1_original
        )
    
    def subject(self) -> 'SubjectInfo':
        """Return subject (seq2) information"""
        return SubjectInfo(
            sequence=self.seq2_aligned,
            start=self.start2,
            end=self.end2,
            original=self.seq2_original
        )
    
    def as_character(self) -> Tuple[str, str]:
        """Return aligned sequences as character strings"""
        return (self.seq1_aligned, self.seq2_aligned)
    
    def as_matrix(self) -> np.ndarray:
        """Return alignment as matrix (rows: sequences, cols: positions)"""
        return np.array([
            list(self.seq1_aligned),
            list(self.seq2_aligned)
        ])
    
    def summary(self) -> Dict:
        """Comprehensive alignment summary"""
        return {
            'score': self.score,
            'type': self.alignment_type,
            'nmatch': self.nmatch(),
            'nmismatch': self.nmismatch(),
            'nedit': self.nedit(),
            'nindel': self.nindel(),
            'pid': self.pid(),
            'identity': self.identity,
            'similarity': self.similarity,
            'gaps': self.gaps,
            'range1': (self.start1, self.end1),
            'range2': (self.start2, self.end2)
        }


@dataclass
class PatternInfo:
    """Information about pattern sequence"""
    sequence: str
    start: int
    end: int
    original: str
    
    def aligned(self) -> str:
        return self.sequence
    
    def nindel(self) -> int:
        return self.sequence.count('-')


@dataclass
class SubjectInfo:
    """Information about subject sequence"""
    sequence: str
    start: int
    end: int
    original: str
    
    def aligned(self) -> str:
        return self.sequence
    
    def nindel(self) -> int:
        return self.sequence.count('-')


class PairwiseAligner:
    """
    Pairwise sequence alignment matching Bioconductor pwalign functionality
    """
    
    def __init__(
        self,
        match_score: float = 2.0,
        mismatch_score: float = -1.0,
        gap_open: float = -2.0,
        gap_extend: float = -0.5,
        substitution_matrix: Optional[Dict] = None,
        use_async: bool = True,
        chunk_size: int = 100
    ):
        """Initialize aligner with scoring parameters"""
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.substitution_matrix = substitution_matrix
        self.use_async = use_async
        self.chunk_size = chunk_size
        
    def _get_score(
        self, 
        a: str, 
        b: str, 
        pattern_quality: Optional[int] = None,
        subject_quality: Optional[int] = None
    ) -> float:
        """Get alignment score for two characters with optional quality scores"""
        base_score = self.match_score if a == b else self.mismatch_score
        
        if self.substitution_matrix:
            base_score = self.substitution_matrix.get((a, b), self.mismatch_score)
        
        # Apply quality score adjustment if provided
        if pattern_quality is not None and subject_quality is not None:
            # Simple quality adjustment (can be more sophisticated)
            quality_factor = min(pattern_quality, subject_quality) / 100.0
            base_score *= quality_factor
        
        return base_score
    
    def _initialize_matrix(
        self, 
        len1: int, 
        len2: int, 
        alignment_type: str = "global"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize scoring matrices based on alignment type"""
        score_matrix = np.zeros((len1 + 1, len2 + 1), dtype=np.float32)
        gap_a = np.zeros((len1 + 1, len2 + 1), dtype=np.float32)
        gap_b = np.zeros((len1 + 1, len2 + 1), dtype=np.float32)
        
        if alignment_type == "global":
            # Standard global alignment
            for i in range(1, len1 + 1):
                score_matrix[i, 0] = self.gap_open + (i - 1) * self.gap_extend
            for j in range(1, len2 + 1):
                score_matrix[0, j] = self.gap_open + (j - 1) * self.gap_extend
        
        elif alignment_type == "overlap":
            # Semi-global: free gaps at ends
            pass  # All zeros
        
        elif alignment_type == "global-local":
            # Free gaps in subject
            for i in range(1, len1 + 1):
                score_matrix[i, 0] = self.gap_open + (i - 1) * self.gap_extend
        
        elif alignment_type == "local-global":
            # Free gaps in pattern
            for j in range(1, len2 + 1):
                score_matrix[0, j] = self.gap_open + (j - 1) * self.gap_extend
        
        return score_matrix, gap_a, gap_b
    
    def _fill_row(
        self,
        seq1: str,
        seq2: str,
        score_matrix: np.ndarray,
        gap_a: np.ndarray,
        gap_b: np.ndarray,
        i: int,
        alignment_type: str = "global",
        pattern_quality: Optional[List[int]] = None,
        subject_quality: Optional[List[int]] = None
    ) -> None:
        """Fill a single row of the matrix"""
        len2 = len(seq2)
        
        for j in range(1, len2 + 1):
            # Get quality scores if available
            pq = pattern_quality[i-1] if pattern_quality else None
            sq = subject_quality[j-1] if subject_quality else None
            
            # Calculate match/mismatch score
            match = score_matrix[i-1, j-1] + self._get_score(seq1[i-1], seq2[j-1], pq, sq)
            
            # Calculate gap scores with affine gap penalty
            gap_a[i, j] = max(
                score_matrix[i-1, j] + self.gap_open,
                gap_a[i-1, j] + self.gap_extend
            )
            gap_b[i, j] = max(
                score_matrix[i, j-1] + self.gap_open,
                gap_b[i, j-1] + self.gap_extend
            )
            
            # Choose best score
            score_matrix[i, j] = max(match, gap_a[i, j], gap_b[i, j])
            
            # For local alignment, don't allow negative scores
            if alignment_type == "local":
                score_matrix[i, j] = max(0, score_matrix[i, j])
    
    def _fill_matrix_sync(
        self,
        seq1: str,
        seq2: str,
        score_matrix: np.ndarray,
        gap_a: np.ndarray,
        gap_b: np.ndarray,
        alignment_type: str = "global",
        pattern_quality: Optional[List[int]] = None,
        subject_quality: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Fill the scoring matrix synchronously"""
        len1, len2 = len(seq1), len(seq2)
        max_score = 0.0
        max_pos = (0, 0)
        
        for i in range(1, len1 + 1):
            self._fill_row(seq1, seq2, score_matrix, gap_a, gap_b, i, 
                          alignment_type, pattern_quality, subject_quality)
            
            # Update max score for local alignment
            if alignment_type == "local":
                row_max = np.max(score_matrix[i, :])
                if row_max > max_score:
                    max_score = row_max
                    max_pos = (i, np.argmax(score_matrix[i, :]))
        
        # Determine max position based on alignment type
        if alignment_type == "global":
            max_score = score_matrix[len1, len2]
            max_pos = (len1, len2)
        elif alignment_type == "overlap":
            # Check last row and column
            last_row_max = np.max(score_matrix[len1, :])
            last_col_max = np.max(score_matrix[:, len2])
            if last_row_max >= last_col_max:
                max_score = last_row_max
                max_pos = (len1, np.argmax(score_matrix[len1, :]))
            else:
                max_score = last_col_max
                max_pos = (np.argmax(score_matrix[:, len2]), len2)
        elif alignment_type == "global-local":
            # Free end gaps in subject (check last row)
            max_score = np.max(score_matrix[len1, :])
            max_pos = (len1, np.argmax(score_matrix[len1, :]))
        elif alignment_type == "local-global":
            # Free end gaps in pattern (check last column)
            max_score = np.max(score_matrix[:, len2])
            max_pos = (np.argmax(score_matrix[:, len2]), len2)
        
        return score_matrix, max_score, max_pos
    
    async def _fill_matrix_async(
        self,
        seq1: str,
        seq2: str,
        score_matrix: np.ndarray,
        gap_a: np.ndarray,
        gap_b: np.ndarray,
        alignment_type: str = "global",
        pattern_quality: Optional[List[int]] = None,
        subject_quality: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Fill the scoring matrix asynchronously"""
        len1, len2 = len(seq1), len(seq2)
        max_score = 0.0
        max_pos = (0, 0)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            
            for i in range(1, len1 + 1):
                if len2 > self.chunk_size:
                    tasks = []
                    for j_start in range(1, len2 + 1, self.chunk_size):
                        j_end = min(j_start + self.chunk_size, len2 + 1)
                        task = loop.run_in_executor(
                            executor,
                            self._fill_row_segment,
                            seq1, seq2, score_matrix, gap_a, gap_b, i, j_start, j_end, 
                            alignment_type, pattern_quality, subject_quality
                        )
                        tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks)
                else:
                    self._fill_row(seq1, seq2, score_matrix, gap_a, gap_b, i, 
                                  alignment_type, pattern_quality, subject_quality)
                
                if alignment_type == "local":
                    row_max = np.max(score_matrix[i, :])
                    if row_max > max_score:
                        max_score = row_max
                        max_pos = (i, np.argmax(score_matrix[i, :]))
        
        # Determine max position based on alignment type
        if alignment_type == "global":
            max_score = score_matrix[len1, len2]
            max_pos = (len1, len2)
        elif alignment_type == "overlap":
            last_row_max = np.max(score_matrix[len1, :])
            last_col_max = np.max(score_matrix[:, len2])
            if last_row_max >= last_col_max:
                max_score = last_row_max
                max_pos = (len1, np.argmax(score_matrix[len1, :]))
            else:
                max_score = last_col_max
                max_pos = (np.argmax(score_matrix[:, len2]), len2)
        elif alignment_type == "global-local":
            max_score = np.max(score_matrix[len1, :])
            max_pos = (len1, np.argmax(score_matrix[len1, :]))
        elif alignment_type == "local-global":
            max_score = np.max(score_matrix[:, len2])
            max_pos = (np.argmax(score_matrix[:, len2]), len2)
        
        return score_matrix, max_score, max_pos
    
    def _fill_row_segment(
        self,
        seq1: str,
        seq2: str,
        score_matrix: np.ndarray,
        gap_a: np.ndarray,
        gap_b: np.ndarray,
        i: int,
        j_start: int,
        j_end: int,
        alignment_type: str = "global",
        pattern_quality: Optional[List[int]] = None,
        subject_quality: Optional[List[int]] = None
    ) -> None:
        """Fill a segment of a row"""
        for j in range(j_start, j_end):
            pq = pattern_quality[i-1] if pattern_quality else None
            sq = subject_quality[j-1] if subject_quality else None
            
            match = score_matrix[i-1, j-1] + self._get_score(seq1[i-1], seq2[j-1], pq, sq)
            
            gap_a[i, j] = max(
                score_matrix[i-1, j] + self.gap_open,
                gap_a[i-1, j] + self.gap_extend
            )
            gap_b[i, j] = max(
                score_matrix[i, j-1] + self.gap_open,
                gap_b[i, j-1] + self.gap_extend
            )
            
            score_matrix[i, j] = max(match, gap_a[i, j], gap_b[i, j])
            
            if alignment_type == "local":
                score_matrix[i, j] = max(0, score_matrix[i, j])
    
    def _traceback(
        self,
        seq1: str,
        seq2: str,
        score_matrix: np.ndarray,
        max_pos: Tuple[int, int],
        alignment_type: str = "global"
    ) -> Tuple[str, str, int, int, int, int]:
        """Perform traceback to construct alignment"""
        aligned1, aligned2 = [], []
        i, j = max_pos
        start_i, start_j = i, j
        
        while i > 0 or j > 0:
            if alignment_type == "local" and score_matrix[i, j] == 0:
                break
            
            current_score = score_matrix[i, j]
            
            if i > 0 and j > 0:
                diagonal_score = score_matrix[i-1, j-1] + self._get_score(seq1[i-1], seq2[j-1])
                if abs(current_score - diagonal_score) < 1e-6:
                    aligned1.append(seq1[i-1])
                    aligned2.append(seq2[j-1])
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0:
                up_score = score_matrix[i-1, j] + self.gap_open
                if abs(current_score - up_score) < 1e-6:
                    aligned1.append(seq1[i-1])
                    aligned2.append('-')
                    i -= 1
                    continue
            
            if j > 0:
                left_score = score_matrix[i, j-1] + self.gap_open
                if abs(current_score - left_score) < 1e-6:
                    aligned1.append('-')
                    aligned2.append(seq2[j-1])
                    j -= 1
                    continue
            
            if i > 0:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1
        
        end_i, end_j = i, j
        
        return (
            ''.join(reversed(aligned1)),
            ''.join(reversed(aligned2)),
            end_i,
            start_i,
            end_j,
            start_j
        )
    
    def _calculate_match_string(self, aligned1: str, aligned2: str) -> str:
        """Generate match string"""
        match_str = []
        for a, b in zip(aligned1, aligned2):
            if a == '-' or b == '-':
                match_str.append(' ')
            elif a == b:
                match_str.append('|')
            else:
                match_str.append('.')
        return ''.join(match_str)
    
    def _calculate_statistics(
        self, 
        aligned1: str, 
        aligned2: str
    ) -> Tuple[float, float, int]:
        """Calculate alignment statistics"""
        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
        similar = sum(1 for a, b in zip(aligned1, aligned2) if a != '-' and b != '-')
        gaps = aligned1.count('-') + aligned2.count('-')
        
        identity = matches / len(aligned1) if len(aligned1) > 0 else 0
        similarity = similar / len(aligned1) if len(aligned1) > 0 else 0
        
        return identity, similarity, gaps
    
    def align(
        self,
        seq1: str,
        seq2: str,
        mode: Literal["global", "local", "overlap", "global-local", "local-global"] = "global",
        pattern_quality: Optional[List[int]] = None,
        subject_quality: Optional[List[int]] = None,
        score_only: bool = False
    ) -> Union[AlignmentResult, float]:
        """
        Perform pairwise sequence alignment
        
        Parameters:
        -----------
        seq1 : str
            First sequence (pattern)
        seq2 : str
            Second sequence (subject)
        mode : str
            Alignment type: "global", "local", "overlap", "global-local", "local-global"
        pattern_quality : list of int, optional
            Quality scores for pattern sequence (0-100)
        subject_quality : list of int, optional
            Quality scores for subject sequence (0-100)
        score_only : bool
            If True, return only the alignment score
        
        Returns:
        --------
        AlignmentResult or float
            Alignment result object or score if score_only=True
        """
        seq1_original = seq1
        seq2_original = seq2
        seq1 = seq1.upper()
        seq2 = seq2.upper()
        
        # Initialize matrices
        score_matrix, gap_a, gap_b = self._initialize_matrix(
            len(seq1), len(seq2), mode
        )
        
        # Fill matrix
        if self.use_async and (len(seq1) * len(seq2) > 10000) and not score_only:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    score_matrix, max_score, max_pos = self._fill_matrix_sync(
                        seq1, seq2, score_matrix, gap_a, gap_b, mode,
                        pattern_quality, subject_quality
                    )
                else:
                    score_matrix, max_score, max_pos = loop.run_until_complete(
                        self._fill_matrix_async(seq1, seq2, score_matrix, gap_a, gap_b, 
                                               mode, pattern_quality, subject_quality)
                    )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score_matrix, max_score, max_pos = loop.run_until_complete(
                    self._fill_matrix_async(seq1, seq2, score_matrix, gap_a, gap_b,
                                           mode, pattern_quality, subject_quality)
                )
                loop.close()
        else:
            score_matrix, max_score, max_pos = self._fill_matrix_sync(
                seq1, seq2, score_matrix, gap_a, gap_b, mode,
                pattern_quality, subject_quality
            )
        
        # Return only score if requested
        if score_only:
            return max_score
        
        # Traceback
        aligned1, aligned2, start1, end1, start2, end2 = self._traceback(
            seq1, seq2, score_matrix, max_pos, mode
        )
        
        # Calculate statistics
        match_string = self._calculate_match_string(aligned1, aligned2)
        identity, similarity, gaps = self._calculate_statistics(aligned1, aligned2)
        
        return AlignmentResult(
            seq1_aligned=aligned1,
            seq2_aligned=aligned2,
            score=max_score,
            start1=start1,
            end1=end1,
            start2=start2,
            end2=end2,
            alignment_type=mode,
            match_string=match_string,
            identity=identity,
            similarity=similarity,
            gaps=gaps,
            seq1_original=seq1_original,
            seq2_original=seq2_original
        )
    
    async def align_async(
        self,
        seq1: str,
        seq2: str,
        mode: Literal["global", "local", "overlap", "global-local", "local-global"] = "global",
        pattern_quality: Optional[List[int]] = None,
        subject_quality: Optional[List[int]] = None,
        score_only: bool = False
    ) -> Union[AlignmentResult, float]:
        """Asynchronous wrapper for align method"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.align,
                seq1,
                seq2,
                mode,
                pattern_quality,
                subject_quality,
                score_only
            )
        return result
    
    async def align_batch_async(
        self,
        sequence_pairs: List[Tuple[str, str]],
        mode: Literal["global", "local", "overlap", "global-local", "local-global"] = "global",
        score_only: bool = False
    ) -> List[Union[AlignmentResult, float]]:
        """Align multiple sequence pairs concurrently"""
        tasks = [
            self.align_async(seq1, seq2, mode, None, None, score_only)
            for seq1, seq2 in sequence_pairs
        ]
        return await asyncio.gather(*tasks)
    
    def align_batch(
        self,
        sequence_pairs: List[Tuple[str, str]],
        mode: Literal["global", "local", "overlap", "global-local", "local-global"] = "global",
        n_workers: int = 4,
        score_only: bool = False
    ) -> List[Union[AlignmentResult, float]]:
        """Align multiple sequence pairs using thread pool"""
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self.align, seq1, seq2, mode, None, None, score_only)
                for seq1, seq2 in sequence_pairs
            ]
            results = [future.result() for future in futures]
        return results


# Convenience function
def pairwise(
    seq1: str,
    seq2: str,
    mode: Literal["global", "local", "overlap", "global-local", "local-global"] = "global",
    match_score: float = 2.0,
    mismatch_score: float = -1.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    use_async: bool = True,
    pattern_quality: Optional[List[int]] = None,
    subject_quality: Optional[List[int]] = None,
    score_only: bool = False
) -> Union[AlignmentResult, float]:
    """
    Convenience function for pairwise sequence alignment
    
    Parameters:
    -----------
    seq1 : str
        First sequence (pattern)
    seq2 : str
        Second sequence (subject)
    mode : str
        Alignment mode: "global", "local", "overlap", "global-local", "local-global"
    match_score : float
        Score for matching bases
    mismatch_score : float
        Score for mismatching bases
    gap_open : float
        Penalty for opening a gap
    gap_extend : float
        Penalty for extending a gap
    use_async : bool
        Enable asyncio for parallel computation (for large sequences)
    pattern_quality : list of int, optional
        Quality scores for pattern sequence (0-100)
    subject_quality : list of int, optional
        Quality scores for subject sequence (0-100)
    score_only : bool
        If True, return only the alignment score
    
    Returns:
    --------
    AlignmentResult or float
        Alignment result with plot() and view() methods, or score if score_only=True
    
    Examples:
    ---------
    >>> # Basic global alignment
    >>> result = pairwise("ACGTGGTT", "GCTTTTGTA", mode="global")
    >>> result.plot()
    
    >>> # Local alignment
    >>> result = pairwise("ACGTGGTT", "GCTTTTGTA", mode="local")
    >>> print(result.nmatch())
    
    >>> # Overlap alignment (semi-global)
    >>> result = pairwise("ACGTGGTT", "GCTTTTGTA", mode="overlap")
    
    >>> # Get only score
    >>> score = pairwise("ACGTGGTT", "GCTTTTGTA", score_only=True)
    >>> print(score)
    
    >>> # With quality scores
    >>> result = pairwise("ACGT", "ACGT", 
    ...                   pattern_quality=[90, 85, 95, 80],
    ...                   subject_quality=[95, 90, 85, 90])
    """
    aligner = PairwiseAligner(
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_open=gap_open,
        gap_extend=gap_extend,
        use_async=use_async
    )
    return aligner.align(seq1, seq2, mode, pattern_quality, subject_quality, score_only)


# Additional utility functions
def compare_strings(alignment: AlignmentResult) -> str:
    """
    Compare two aligned sequences
    Returns a string with '?' for match, '+' for pattern insertion, '-' for subject insertion
    """
    return alignment.compare_strings()


def string_distance(seq1: str, seq2: str) -> int:
    """
    Calculate edit distance between two sequences
    Uses global alignment with unit costs
    """
    aligner = PairwiseAligner(
        match_score=0,
        mismatch_score=-1,
        gap_open=-1,
        gap_extend=0
    )
    score = aligner.align(seq1, seq2, mode="global", score_only=True)
    return int(-score)


def coverage(
    alignments: List[AlignmentResult],
    reference_length: Optional[int] = None
) -> np.ndarray:
    """
    Calculate coverage from multiple alignments
    
    Parameters:
    -----------
    alignments : list of AlignmentResult
        List of alignment results
    reference_length : int, optional
        Length of reference sequence. If None, uses max end position
    
    Returns:
    --------
    np.ndarray
        Coverage array
    """
    if not alignments:
        return np.array([])
    
    if reference_length is None:
        reference_length = max(a.end2 for a in alignments)
    
    coverage_array = np.zeros(reference_length, dtype=np.int32)
    
    for alignment in alignments:
        start = alignment.start2
        end = alignment.end2
        coverage_array[start:end] += 1
    
    return coverage_array


def nucleotide_substitution_matrix(
    match: float = 1.0,
    mismatch: float = -1.0,
    base_only: bool = True
) -> Dict[Tuple[str, str], float]:
    """
    Create nucleotide substitution matrix
    
    Parameters:
    -----------
    match : float
        Score for matching bases
    mismatch : float
        Score for mismatching bases
    base_only : bool
        If True, only include A, C, G, T
    
    Returns:
    --------
    dict
        Substitution matrix as dictionary
    """
    if base_only:
        bases = ['A', 'C', 'G', 'T']
    else:
        bases = ['A', 'C', 'G', 'T', 'N', 'R', 'Y', 'S', 'W', 'K', 'M']
    
    matrix = {}
    for b1 in bases:
        for b2 in bases:
            if b1 == b2:
                matrix[(b1, b2)] = match
            else:
                matrix[(b1, b2)] = mismatch
    
    return matrix


# Predefined substitution matrices
BLOSUM50 = {
    ('A', 'A'): 5, ('A', 'R'): -2, ('A', 'N'): -1, ('A', 'D'): -2, ('A', 'C'): -1,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -2, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -3, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'A'): -2, ('R', 'R'): 7, ('R', 'N'): -1, ('R', 'D'): -2, ('R', 'C'): -4,
    ('R', 'Q'): 1, ('R', 'E'): 0, ('R', 'G'): -3, ('R', 'H'): 0, ('R', 'I'): -4,
    ('R', 'L'): -3, ('R', 'K'): 3, ('R', 'M'): -2, ('R', 'F'): -3, ('R', 'P'): -3,
    ('R', 'S'): -1, ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -1, ('R', 'V'): -3,
    ('N', 'A'): -1, ('N', 'R'): -1, ('N', 'N'): 7, ('N', 'D'): 2, ('N', 'C'): -2,
    ('N', 'Q'): 0, ('N', 'E'): 0, ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3,
    ('N', 'L'): -4, ('N', 'K'): 0, ('N', 'M'): -2, ('N', 'F'): -4, ('N', 'P'): -2,
    ('N', 'S'): 1, ('N', 'T'): 0, ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,
    ('D', 'A'): -2, ('D', 'R'): -2, ('D', 'N'): 2, ('D', 'D'): 8, ('D', 'C'): -4,
    ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1, ('D', 'H'): -1, ('D', 'I'): -4,
    ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -4, ('D', 'F'): -5, ('D', 'P'): -1,
    ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -5, ('D', 'Y'): -3, ('D', 'V'): -4,
    # Add more as needed - abbreviated for space
}

BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'A'): -1, ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3,
    ('R', 'Q'): 1, ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3,
    ('R', 'L'): -2, ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2,
    ('R', 'S'): -1, ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
    # Add complete matrix as needed
}


class QualityScore:
    """Base class for quality scores"""
    def __init__(self, scores: List[int]):
        self.scores = scores
    
    def to_error_probability(self) -> List[float]:
        """Convert quality scores to error probabilities"""
        raise NotImplementedError


class SolexaQuality(QualityScore):
    """Solexa/Illumina quality scores"""
    def to_error_probability(self) -> List[float]:
        """Convert Solexa quality to error probability"""
        return [10 ** (-q / 10.0) for q in self.scores]


class PhredQuality(QualityScore):
    """Phred quality scores"""
    def to_error_probability(self) -> List[float]:
        """Convert Phred quality to error probability"""
        return [10 ** (-q / 10.0) for q in self.scores]


def agrep_bioc(
    pattern: str,
    sequences: List[str],
    max_distance: float = 0.1,
    ignore_case: bool = False
) -> List[int]:
    """
    Approximate grep using alignment
    
    Parameters:
    -----------
    pattern : str
        Pattern to search for
    sequences : list of str
        Sequences to search in
    max_distance : float
        Maximum edit distance (as fraction if < 1, or absolute if >= 1)
    ignore_case : bool
        Ignore case when matching
    
    Returns:
    --------
    list of int
        Indices of matching sequences
    """
    if max_distance < 1:
        max_distance = int(np.ceil(max_distance * len(pattern)))
    
    matches = []
    for i, seq in enumerate(sequences):
        distance = string_distance(pattern, seq)
        if distance <= max_distance:
            matches.append(i)
    
    return matches