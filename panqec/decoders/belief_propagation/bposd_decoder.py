import numpy as np
from ldpc import bp_decoder, BpOsdDecoder
from ldpc.bp_flip import BpFlipDecoder
from panqec.codes import StabilizerCode
from panqec.decoders.base.flip import FlipDecoder
from panqec.decoders.base.flip_modified import RandFlip

from panqec.error_models import BaseErrorModel
from panqec.decoders import BaseDecoder

from typing import Optional, List, Union

"""
    Belief propagation and Ordered Statistic Decoding (OSD) decoder for binary linear codes.

    This class provides an implementation of the BP decoding that uses Ordered Statistic Decoding (OSD)
    as a fallback method if the BP does not converge. The class inherits from the `BpDecoderBase` class.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix for the code.
    error_rate : Optional[float], optional
        The probability of a bit being flipped in the received codeword, by default None.
    error_channel : Optional[List[float]], optional
        A list of probabilities that specify the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code, by default None.
    max_iter : Optional[int], optional
        The maximum number of iterations for the decoding algorithm, by default 0.
    bp_method : Optional[str], optional
        The belief propagation method used. Must be one of {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor used in the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method used. Must be one of {'parallel', 'serial'}, by default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads used for parallel decoding, by default 1.
    random_schedule_seed : Optional[int], optional
        Whether to use a random serial schedule order, by default False.
    serial_schedule_order : Optional[List[int]], optional
        A list of integers that specify the serial schedule order. Must be of length equal to the block length of the code,
        by default None.
    osd_method : int, optional
        The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}.
    osd_order : int, optional
        The OSD order, by default 0.
    __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, osd_method: Union[str, int, float] = 0,
                 osd_order: int = 0, input_vector_type: str = "syndrome", **kwargs):
"""
class BeliefPropagationOSDDecoder(BaseDecoder):
    label = 'BP-OSD decoder'
    allowed_codes = None  # all codes allowed

    def __init__(self, code: StabilizerCode, error_model: BaseErrorModel, error_rate: float,
                 max_bp_iter: int = 20, osd_order: int = 0, bp_method: str = 'minimum_sum',
                 error_channel: Optional[List[float]] = None, ms_scaling_factor: Optional[float] = 0.75,
                 schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,   channel_update: bool = False,
                 random_serial_schedule: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None,
                 osd_method: Union[str, int, float] = 'OSD_CS', flip_max_iter: int = 6, 
                 flip_pfreq: int = 2, flip_seed: Optional[int] = 32, rand_bp:bool=False, rand_flip:bool=False, 
                 ldpc_bp_flip:bool=False, bp_osd:bool=False, rand_shuffle:bool=False, p_flip:bool=False, p_bp:bool =False):
        
        super().__init__(code, error_model, error_rate)
        self._max_bp_iter = max_bp_iter
        self._osd_order = osd_order
        self._bp_method = bp_method
        self._error_channel = error_channel
        self._ms_scaling_factor = ms_scaling_factor
        self._schedule = schedule
        self._omp_thread_count = omp_thread_count
        self._random_serial_schedule = random_serial_schedule
        self._serial_schedule_order = serial_schedule_order
        self._osd_method = osd_method
        self._channel_update = channel_update
        # Initialize the FlipDecoder with specified parameters
        self.flip_max_iter = flip_max_iter
        self.flip_pfreq = flip_pfreq
        self.flip_seed = flip_seed
        self._initialized = False
        self._rand_bp = rand_bp
        self._ldpc_bp_flip = ldpc_bp_flip
        self._rand_flip = rand_flip
        self._bp_osd = bp_osd
        self._rand_shuffle = rand_shuffle
        self._p_flip = p_flip
        self._p_bp = p_bp

        true_flags = sum([self._rand_bp, self._ldpc_bp_flip, self._rand_flip, self._bp_osd, self._p_flip, self._p_bp])
        if true_flags > 1:
            raise ValueError("Only one of rand_bp, ldpc_bp_flip, rand_flip, bp_osd, p_flip, or p_bp can be True at a time.")



    @property
    def params(self) -> dict:
        return {
            'max_bp_iter': self._max_bp_iter,
            'channel_update': self._channel_update,
            'osd_order': self._osd_order,
            'bp_method': self._bp_method
        }

    def get_probabilities(self):
        pi, px, py, pz = self.error_model.probability_distribution(
            self.code, self.error_rate
        )

        return pi, px, py, pz

    def update_probabilities(self, correction: np.ndarray,
                             px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                             direction: str = "x->z") -> np.ndarray:
        """Update X probabilities once a Z correction has been applied"""

        n_qubits = correction.shape[0]

        new_probs = np.zeros(n_qubits)

        if direction == "z->x":
            for i in range(n_qubits):
                if correction[i] == 1:
                    if pz[i] + py[i] != 0:
                        new_probs[i] = py[i] / (pz[i] + py[i])
                else:
                    new_probs[i] = px[i] / (1 - pz[i] - py[i])

        elif direction == "x->z":
            for i in range(n_qubits):
                if correction[i] == 1:
                    if px[i] + py[i] != 0:
                        new_probs[i] = py[i] / (px[i] + py[i])
                else:
                    new_probs[i] = pz[i] / (1 - px[i] - py[i])

        else:
            raise ValueError(
                f"Unrecognized direction {direction} when "
                "updating probabilities"
            )

        return new_probs

    def initialize_decoders(self):
        is_css = self.code.is_css

        if is_css:
            if not self._ldpc_bp_flip:
                self.z_decoder = bp_decoder(
                    self.code.Hx,
                    error_rate=self.error_rate,
                    max_iter=self._max_bp_iter,
                    bp_method=self._bp_method,
                    ms_scaling_factor = self._ms_scaling_factor,
                    input_vector_type='syndrome', 
                    #schedule = self._schedule,
                    #omp_thread_count = self._omp_thread_count,
                    #random_schedule_seed = self._random_serial_schedule,
                    #serial_schedule_order = self._serial_schedule_order,
                    #osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                    #osd_order=self._osd_order
                )
                    

                self.x_decoder = bp_decoder(
                    self.code.Hz,
                    error_rate=self.error_rate,
                    max_iter=self._max_bp_iter,
                    bp_method=self._bp_method,
                    ms_scaling_factor = self._ms_scaling_factor,
                    input_vector_type='syndrome', 
                    #schedule = self._schedule,
                    #omp_thread_count = self._omp_thread_count,
                    #random_serial_schedule = self._random_serial_schedule,
                    #serial_schedule_order = self._serial_schedule_order,
                    #osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                    #osd_order=self._osd_order
                )
            if self._ldpc_bp_flip:
                print("Utilising p-BP from LDPC library")
                self.z_decoder = BpFlipDecoder(
                    self.code.Hx,
                    error_rate=self.error_rate,
                    max_iter=self._max_bp_iter,
                    bp_method=self._bp_method,
                    ms_scaling_factor = self._ms_scaling_factor,
                    flip_iterations = self.flip_max_iter,
                    pflip_frequency = self.flip_pfreq,
                    pflip_seed = 32,
                    #input_vector_type='syndrome', 
                    #schedule = self._schedule,
                    #omp_thread_count = self._omp_thread_count,
                    random_schedule_seed = 30,
                    #serial_schedule_order = self._serial_schedule_order,
                    #osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                    #osd_order=self._osd_order
                )
                    

                self.x_decoder = BpFlipDecoder(
                    self.code.Hz,
                    error_rate=self.error_rate,
                    max_iter=self._max_bp_iter,
                    bp_method=self._bp_method,
                    ms_scaling_factor = self._ms_scaling_factor,
                    flip_iterations = self.flip_max_iter,
                    pflip_frequency = self.flip_pfreq,
                    pflip_seed = 32,
                    #input_vector_type='syndrome', 
                    #schedule = self._schedule,
                    #omp_thread_count = self._omp_thread_count,
                    random_schedule_seed = 30,
                    #serial_schedule_order = self._serial_schedule_order,
                    #osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                    #osd_order=self._osd_order
                )

            

        else:
            self.decoder = bp_decoder(
                self.code.stabilizer_matrix,
                error_rate=self.error_rate,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                input_vector_type='syndrome', 
                #ms_scaling_factor = self._ms_scaling_factor,
                #schedule = self._schedule,
                #omp_thread_count = self._omp_thread_count,
                #random_serial_schedule = self._random_serial_schedule,
                #serial_schedule_order = self._serial_schedule_order,
                #osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                #osd_order=self._osd_order
            )
        self._initialized = True

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self.initialize_decoders()

        is_css = self.code.is_css
        n_qubits = self.code.n
        syndrome = np.array(syndrome, dtype=int)

        if is_css:
            syndrome_z = self.code.extract_z_syndrome(syndrome)
            syndrome_x = self.code.extract_x_syndrome(syndrome)
            original_syndrome_z = syndrome_z.copy()
            original_syndrome_x = syndrome_x.copy()
            flip_syndrome_z = syndrome_z.copy()
            flip_syndrome_x = syndrome_x.copy()
           

        pi, px, py, pz = self.get_probabilities()

        probabilities_x = px + py
        probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if is_css:

            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)
            
            if self._p_flip:
    
                flip_x_decoder  = FlipDecoder(self.code.Hz, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed)
                flip_correction_x = flip_x_decoder.decode(flip_syndrome_z)
            

                flip_z_decoder = FlipDecoder(self.code.Hx, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed)
                flip_correction_z = flip_z_decoder.decode(flip_syndrome_x)

                correction = np.concatenate([flip_correction_x, flip_correction_z])

         
                return correction


            if self._rand_flip:
    
                flip_x_decoder  = RandFlip(self.code.Hz, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed, shuffle=self._rand_shuffle)
                flip_correction_x = flip_x_decoder.decode(flip_syndrome_z)
            

                flip_z_decoder = RandFlip(self.code.Hx, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed, shuffle=self._rand_shuffle)
                flip_correction_z = flip_z_decoder.decode(flip_syndrome_x)

                correction = np.concatenate([flip_correction_x, flip_correction_z])

         
                return correction
            
            if self._p_bp:
                self.x_decoder.decode(original_syndrome_z)
                x_correction = self.x_decoder.decoding
                resulting_syndrome_z = np.matmul(self.code.Hz.toarray(), x_correction) % 2 
                new_syndrome_z = np.logical_xor(syndrome_z, resulting_syndrome_z)
                flip_x_decoder = FlipDecoder(self.code.Hz, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed)
                flip_correction_x = flip_x_decoder.decode(new_syndrome_z)
                final_x_correction = np.logical_xor(x_correction,flip_correction_x) % 2


                self.z_decoder.decode(original_syndrome_x)
                z_correction = self.z_decoder.decoding
                resulting_syndrome_x = np.matmul(self.code.Hx.toarray(), z_correction) % 2 
                new_syndrome_x = np.logical_xor(syndrome_x, resulting_syndrome_x)
                flip_z_decoder = FlipDecoder(self.code.Hx, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed)
                flip_correction_z = flip_z_decoder.decode(new_syndrome_x)
                final_z_correction = np.logical_xor(z_correction,flip_correction_z) % 2

                correction = np.concatenate([final_x_correction, final_z_correction])
                return correction

         
            if self._rand_bp:

                self.x_decoder.decode(original_syndrome_z)
                x_correction = self.x_decoder.decoding
                resulting_syndrome_z = np.matmul(self.code.Hz.toarray(), x_correction) % 2 
                new_syndrome_z = np.logical_xor(syndrome_z, resulting_syndrome_z)
                flip_x_decoder = RandFlip(self.code.Hz, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed, shuffle=self._rand_shuffle)
                flip_correction_x = flip_x_decoder.decode(new_syndrome_z)
                final_x_correction = np.logical_xor(x_correction,flip_correction_x) % 2


                self.z_decoder.decode(original_syndrome_x)
                z_correction = self.z_decoder.decoding
                resulting_syndrome_x = np.matmul(self.code.Hx.toarray(), z_correction) % 2 
                new_syndrome_x = np.logical_xor(syndrome_x, resulting_syndrome_x)
                flip_z_decoder = RandFlip(self.code.Hx, max_iter=self.flip_max_iter, pfreq=self.flip_pfreq, seed=self.flip_seed,shuffle=self._rand_shuffle)
                flip_correction_z = flip_z_decoder.decode(new_syndrome_x)
                final_z_correction = np.logical_xor(z_correction,flip_correction_z) % 2

                correction = np.concatenate([final_x_correction, final_z_correction])
         
                return correction
            
            if self._ldpc_bp_flip:

                z_correction = self.z_decoder.decode(syndrome_x)

            # Bayes update of the probability
                if self._channel_update:
                    new_x_probs = self.update_probabilities(
                        z_correction, px, py, pz, direction="z->x"
                    )
                
                x_correction = self.x_decoder.decode(syndrome_z)
            
                correction = np.concatenate([x_correction, z_correction])

                return correction

           
            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)

            self.z_decoder.decode(syndrome_x)
            z_correction = self.z_decoder.decoding

            if self._channel_update:
                new_x_probs = self.update_probabilities(
                    z_correction, px, py, pz, direction="z->x"
                )
                self.x_decoder.update_channel_probs(new_x_probs)

            self.x_decoder.decode(syndrome_z)
            x_correction = self.x_decoder.decoding

            if self._bp_osd:
                    x_osd_decoder = BpOsdDecoder(
                        self.code.Hz,
                        error_rate=self.error_rate,
                        max_iter=0,
                        bp_method=self._bp_method,
                        ms_scaling_factor = self._ms_scaling_factor,
                        #input_vector_type='syndrome', 
                        #schedule = self._schedule,
                        #omp_thread_count = self._omp_thread_count,
                        #random_schedule_seed = self._random_serial_schedule,
                        #serial_schedule_order = self._serial_schedule_order,
                        osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                        osd_order=self._osd_order
                    )

                    resulting_bp_syndrome_z = np.matmul(self.code.Hz.toarray(), x_correction) % 2 
                    new_syndrome_z = np.logical_xor(syndrome_z, resulting_bp_syndrome_z)
                    
                    x_osd_correction = x_osd_decoder.decode(new_syndrome_z)
                    x_correction = np.logical_xor(x_correction ,x_osd_correction) % 2

                    z_osd_decoder = BpOsdDecoder(
                        self.code.Hx,
                        error_rate=self.error_rate,
                        max_iter=0,
                        bp_method=self._bp_method,
                        ms_scaling_factor = self._ms_scaling_factor,
                        #input_vector_type='syndrome', 
                        #schedule = self._schedule,
                        #omp_thread_count = self._omp_thread_count,
                        #random_schedule_seed = self._random_serial_schedule,
                        #serial_schedule_order = self._serial_schedule_order,
                        osd_method=self._osd_method,  #  The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}
                        osd_order=self._osd_order
                    )

                    resulting_bp_syndrome_x = np.matmul(self.code.Hx.toarray(), z_correction) % 2 
                    new_syndrome_x = np.logical_xor(syndrome_x, resulting_bp_syndrome_x)
                    
                    z_osd_correction = z_osd_decoder.decode(new_syndrome_x)
                    z_correction = np.logical_xor(z_correction ,z_osd_correction) % 2
            
            correction = np.concatenate([x_correction, z_correction])

            return correction
        
        else:
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.decoder.update_channel_probs(probabilities)

            # Decode all errors
            self.decoder.decode(syndrome)
            correction = self.decoder.osdw_decoding
            correction = np.concatenate(
                [correction[n_qubits:], correction[:n_qubits]]
            )

            return correction


            

        
    def old_decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        if not self._initialized:
            self.initialize_decoders()

        is_css = self.code.is_css
        n_qubits = self.code.n
        syndrome = np.array(syndrome, dtype=int)

        if is_css:
            syndrome_z = self.code.extract_z_syndrome(syndrome)
            syndrome_x = self.code.extract_x_syndrome(syndrome)

        pi, px, py, pz = self.get_probabilities()

        probabilities_x = px + py
        probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if is_css:
            print("")
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)

            # Decode Z errors
            self.z_decoder.decode(syndrome_x)
            z_correction = self.z_decoder.osdw_decoding

            # Bayes update of the probability
            if self._channel_update:
                new_x_probs = self.update_probabilities(
                    z_correction, px, py, pz, direction="z->x"
                )
                self.x_decoder.update_channel_probs(new_x_probs)

            # Decode X errors
            self.x_decoder.decode(syndrome_z)
            x_correction = self.x_decoder.osdw_decoding
            
            correction = np.concatenate([x_correction, z_correction])

        else:
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.decoder.update_channel_probs(probabilities)

            # Decode all errors
            self.decoder.decode(syndrome)
            correction = self.decoder.osdw_decoding
            correction = np.concatenate(
                [correction[n_qubits:], correction[:n_qubits]]
            )

        return correction


def test_decoder():
    from panqec.codes import XCubeCode
    from panqec.error_models import PauliErrorModel
    import time
    rng = np.random.default_rng()

    L = 20
    code = XCubeCode(L, L, L)

    error_rate = 0.5
    r_x, r_y, r_z = [0.15, 0.15, 0.7]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    print("Create stabilizer matrix")
    code.stabilizer_matrix

    print("Create Hx and Hz")
    code.Hx
    code.Hz

    print("Create logicals")
    code.logicals_x
    code.logicals_z

    print("Instantiate BP-OSD")
    decoder = BeliefPropagationOSDDecoder(
        code, error_model, error_rate, osd_order=0, max_bp_iter=1000
    )

    # Start timer
    start = time.time()

    n_iter = 1
    accuracy = 0
    for i in range(n_iter):
        print(f"\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, error_rate, rng=rng)
        print("Calculate syndrome")
        syndrome = code.measure_syndrome(error)
        print("Decode")
        correction = decoder.decode(syndrome)
        print("Get total error")
        total_error = (correction + error) % 2

        codespace = code.in_codespace(total_error)
        success = not code.is_logical_error(total_error) and codespace
        print(success)
        accuracy += success

    accuracy /= n_iter
    print("Average time per iteration", (time.time() - start) / n_iter)
    print("Logical error rate", 1 - accuracy)


if __name__ == '__main__':
    test_decoder()
