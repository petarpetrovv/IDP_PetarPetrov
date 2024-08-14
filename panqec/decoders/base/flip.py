import numpy as np

from typing import Optional, List, Union
from typing import Optional, List
from scipy.sparse import csr_matrix
from typing import Optional
class FlipDecoder:

    def __init__(self, pcm: csr_matrix, max_iter: int = 6, pfreq: int = 2, seed: Optional[int] = None, memory:bool = False):
        self.pcm = pcm
        self.max_iter = max_iter 
        self.pfreq = pfreq 
        self.seed = seed
        self.RNG = np.random.default_rng(seed)
        self.bit_count = self.pcm.shape[1]
        self.check_count = self.pcm.shape[0]
        self.decoding = np.zeros(self.bit_count, dtype=np.uint8)
        self.memory = memory

    def decode(self, syndrome: np.ndarray):
        decoding = np.zeros_like(self.decoding, dtype=np.int64)  
        current_syndrome = syndrome.astype(np.int64).copy()  

        syndrome_history = []

        
        if np.sum(current_syndrome) == 0:
                        return decoding
    
        pcm_t = self.pcm.transpose().tocsr()  
        for iteration in range(1, self.max_iter + 1):

            for bit_idx in range(self.bit_count):
                bit_flip = False
                checks = pcm_t.indices[pcm_t.indptr[bit_idx]:pcm_t.indptr[bit_idx + 1]]
                unsatisfied_checks = current_syndrome[checks] == 1
                satisfied_checks = ~unsatisfied_checks

                if np.sum(satisfied_checks) <  np.sum(unsatisfied_checks):
                    bit_flip = True

                elif iteration % self.pfreq == 0 and np.sum(satisfied_checks) == np.sum(unsatisfied_checks):
                    if self.RNG.random() < 0.5:
                        bit_flip = True

                if bit_flip:
                        decoding[bit_idx] ^= 1
                        current_syndrome[checks] ^= 1
                        if np.sum(current_syndrome) == 0:
                            return decoding
            
            if len(syndrome_history) >= 3 and np.array_equal(syndrome_history[-1], current_syndrome) and self.memory:
                candidates = []
                for bit_idx in range(self.bit_count):
                    checks = pcm_t.indices[pcm_t.indptr[bit_idx]:pcm_t.indptr[bit_idx + 1]]
                    unsatisfied_checks = current_syndrome[checks] == 1
                    satisfied_checks = ~unsatisfied_checks
                    if np.sum(unsatisfied_checks) == np.sum(satisfied_checks):
                        candidates.append(bit_idx)

                if candidates:
                    chosen_bit = self.RNG.choice(candidates)
                    checks = pcm_t.indices[pcm_t.indptr[chosen_bit]:pcm_t.indptr[chosen_bit + 1]]
                    decoding[chosen_bit] ^= 1
                    current_syndrome[checks] ^= 1
                    if np.sum(current_syndrome) == 0:
                            return decoding
            
            
            syndrome_history.append(current_syndrome.copy())
            if len(syndrome_history) > 5:
                syndrome_history.pop(0)
        
        return decoding

                    

if __name__ == "__main__":
    pcm = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    decoder = FlipDecoder(pcm, max_iter=10, pfreq=10, seed=42)
    syndrome = np.array([1, 1, 0], dtype=np.uint8)
    result = decoder.decode(syndrome)
    print("Decoding result:", result)

