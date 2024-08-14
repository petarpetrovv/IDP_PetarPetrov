import numpy as np
import math
from typing import Optional, List, Union
from typing import Optional, List
from scipy.sparse import csr_matrix
from typing import Optional

class RandFlip:

    def __init__(self, pcm: csr_matrix, max_iter: int = 6, pfreq: int = 2, seed: Optional[int] = None, shuffle: bool = False):
        self.pcm = pcm
        self.max_iter = max_iter 
        self.pfreq = pfreq 
        self.seed = seed
        self.shuffle = shuffle
        self.RNG = np.random.default_rng(32)
        self.bit_count = self.pcm.shape[1]
        self.check_count = self.pcm.shape[0]
        self.decoding = np.zeros(self.bit_count, dtype=np.uint8)
        self.indices = list(range(self.bit_count))
        if self.shuffle:
            self.RNG.shuffle(self.indices)


    def decode(self, syndrome: np.ndarray):
        decoding = np.zeros_like(self.decoding, dtype=np.int64)  
        current_syndrome = syndrome.astype(np.int64).copy()  
        
        if np.sum(current_syndrome) == 0:
            return decoding

        pcm_t = self.pcm.transpose().tocsr()
        syndrome_history = []
        flip_probability = 0.5
        #flip_probability = 1 / (self.bit_count + self.check_count)
        #flip_probability =  max(0.25, 0.5 - (self.bit_count + self.check_count) / 1000.0)

        
        for iteration in range(1, self.max_iter + 1):

            for bit_idx in self.indices:
                    checks = pcm_t.indices[pcm_t.indptr[bit_idx]:pcm_t.indptr[bit_idx + 1]]
                    unsatisfied_checks = current_syndrome[checks] == 1
                    satisfied_checks = ~unsatisfied_checks

                    if np.sum(satisfied_checks) == 0 and np.sum(unsatisfied_checks) == len(checks):
                        decoding[bit_idx] ^= 1
                        current_syndrome[checks] ^= 1

                    if np.sum(current_syndrome) == 0:
                        return decoding
            
            candidates = []
            for bit_idx in self.indices:
                    checks = pcm_t.indices[pcm_t.indptr[bit_idx]:pcm_t.indptr[bit_idx + 1]]
                    unsatisfied_checks = current_syndrome[checks] == 1
                    satisfied_checks = ~unsatisfied_checks

                    if np.sum(satisfied_checks) < np.sum(unsatisfied_checks):
                        decoding[bit_idx] ^= 1
                        current_syndrome[checks] ^= 1
                        if np.sum(current_syndrome) == 0:
                            return decoding
                        else:
                            continue
                    
                    elif np.sum(unsatisfied_checks) == np.sum(satisfied_checks):
                        candidates.append(bit_idx)

            
            if iteration % self.pfreq == 0 and len(candidates) != 0:
                chosen_bit = self.RNG.choice(candidates)
                checks = pcm_t.indices[pcm_t.indptr[chosen_bit]:pcm_t.indptr[chosen_bit + 1]]
                decoding[chosen_bit] ^= 1
                current_syndrome[checks] ^= 1
                if np.sum(current_syndrome) == 0:
                            return decoding


            if len(syndrome_history) >= 3 and np.array_equal(syndrome_history[-1], current_syndrome):
                candidates = []
                for bit_idx in self.indices:
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


                    

