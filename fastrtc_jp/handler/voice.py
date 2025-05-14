import numpy as np
import typing
if typing.TYPE_CHECKING:
    from fastrtc_jp.handler.agent_task import AgentTask

def str_join(a:str|None,b:str|None,sep:str=""):
    if a is not None:
        if len(a)>0:
            if b is not None and len(b)>0:
                return a+sep+b
            return a
        else:
            if b is None:
                return a
    return b

class SttAudio:
    """
    Class for handling speech-to-text audio data and its transcription.

    This class manages audio segments from speech input, including their timing information,
    and provides functionality for combining consecutive audio segments.

    Attributes:
        s (int): The start sample index of the audio segment.
        e (int): The end sample index of the audio segment.
        rate (int): The sampling rate of the audio in Hz.
        audio (np.ndarray): The audio data as a 2D numpy array of shape (1, N).
        user_input (str | None): The text transcription of the audio segment, 
            None if not yet transcribed.
    """

    def __init__(self, s:int, e:int, rate:int, audio:np.ndarray):
        """
        Initializes a voice object with the given parameters.
        
        Args:
            s (int): The start sample index of the audio segment.
            e (int): The end sample index of the audio segment.
            rate (int): The sampling rate of the audio in Hz.
            audio (np.ndarray): A 2D numpy array representing the audio data. 
                Must be of shape (1, N) and type int16 or float32.
        
        Raises:
            ValueError: If the audio is not a 2D array of shape (1, N) with type int16 or float32.
        
        Attributes:
            s (int): The start sample index of the audio segment.
            e (int): The end sample index of the audio segment.
            rate (int): The sampling rate of the audio in Hz.
            audio (np.ndarray): The audio data as a 2D numpy array.
            user_input (str | None): The text transcription of the audio segment, None if not yet transcribed.
        """

        if audio.dtype != np.int16 and audio.dtype != np.float32 or len(audio.shape)!=2 or audio.shape[0]!=1:
            raise ValueError("Audio must be a 2D array of shape (1, N) with type int16 or float32")
        self.s:int = s
        self.e:int = e
        self.rate:int = rate
        self.audio:np.ndarray = audio
        # result of speech-to-text
        self.user_input:str|None = None

    def append(self, stt_audio:"SttAudio"):
        """
        Appends the audio data from another SttAudio object to the current object.

        Parameters:
            stt_audio (SttAudio): The SttAudio object whose audio data will be appended.

        Raises:
            ValueError: If the sample rates of the two audio objects do not match.
            ValueError: If the audio data types of the two audio objects do not match.

        Notes:
            - If there is an overlap between the end of the current audio and the start of the 
              provided audio, the overlapping portion of the provided audio is excluded.
            - Updates the end time (`e`) of the current audio to match the end time of the 
              appended audio.
            - If the provided audio contains user input, it is appended to the current 
              object's user input. If the current object has no user input, it is initialized 
              with the provided audio's user input.
        """
        if self.rate != stt_audio.rate:
            raise ValueError("Sample rates must match to append audio.")
        if self.audio.dtype != stt_audio.audio.dtype:
            raise ValueError("Audio data types must match to append audio.")
        overlap = max(0, self.e - stt_audio.s)
        if overlap > 0:
            self.audio = np.concatenate((self.audio, stt_audio.audio[:, overlap:]), axis=1)
        else:
            self.audio = np.concatenate((self.audio, stt_audio.audio), axis=1)
        self.e = stt_audio.e
        if stt_audio.user_input:
            if self.user_input:
                self.user_input += stt_audio.user_input
            else:
                self.user_input = stt_audio.user_input

    def _extend_buffer(self, s:int, e:int):
        """
        Extend the audio buffer with zeros if needed.
        
        Args:
            s (int): Start sample index.
            e (int): End sample index.
        """
        if s < self.s:
            # Add zeros at the beginning
            prefix_length = self.s - s
            prefix = np.zeros((1, prefix_length), dtype=self.audio.dtype)
            self.audio = np.concatenate((prefix, self.audio), axis=1)
            self.s = s
            
        if e > self.e:
            # Add zeros at the end
            suffix_length = e - self.e
            suffix = np.zeros((1, suffix_length), dtype=self.audio.dtype)
            self.audio = np.concatenate((self.audio, suffix), axis=1)
            self.e = e

    def _join_text(self, s:int, user_input:str|None):
        """
        Join text based on temporal order.
        
        Args:
            s (int): Start sample index of the new text.
            user_input (str | None): Text to join.
        """
        if s < self.s:
            self.user_input = str_join(user_input, self.user_input)
        else:
            self.user_input = str_join(self.user_input, user_input)

    def append_text(self, s:int, e:int, user_input:str|None):
        """
        Append text and extend the audio buffer with zeros as needed.
        
        This method updates the text transcription and adjusts the audio buffer based on
        the timing information. The audio buffer is extended with zeros to accommodate
        the new time range.
        
        Args:
            s (int): Start sample index of the new text segment.
            e (int): End sample index of the new text segment.
            user_input (str | None): The text to append.
            
        Note:
            The method will:
            1. Join text based on temporal order (prepend or append)
            2. Extend audio buffer with zeros if the new segment extends beyond current bounds
            3. Update start and end indices (s, e) to reflect the new time range
        """
        self._join_text(s, user_input)
        self._extend_buffer(s, e)

    def append_audio(self, s:int, audio:np.ndarray, user_input:str|None):
        """
        Append audio data and its associated text at a specific time point.
        
        Similar to append() but accepts individual components instead of an SttAudio object.
        
        Args:
            s (int): Start sample index of the audio segment.
            audio (np.ndarray): Audio data as a 2D numpy array of shape (1, N).
                Must be of type int16 or float32.
            user_input (str | None): Associated text transcription, if any.
            
        Raises:
            ValueError: If the audio data types do not match.
            ValueError: If the audio is not a 2D array of shape (1, N).
        """
        # Validate audio format
        if audio.dtype != self.audio.dtype:
            raise ValueError("Audio data types must match.")
        if len(audio.shape) != 2 or audio.shape[0] != 1:
            raise ValueError("Audio must be a 2D array of shape (1, N)")
        
        # Calculate new end point
        e = s + audio.shape[1]
        
        self._join_text(s, user_input)
        self._extend_buffer(s, e)
        
        # Insert the new audio data at the correct position
        start_idx = s - self.s
        end_idx = start_idx + audio.shape[1]
        self.audio[0, start_idx:end_idx] = audio[0]

class SttAudioBuffer:
    def __init__(self):
        self.audio_list:list[SttAudio] = []

    def __len__(self) ->int:
        return len(self.audio_list)

    def copy_to_list(self) ->list[SttAudio]:
        return [s for s in self.audio_list]

    def reset(self):
        self.audio_list = []

    def append(self,audio:SttAudio):
        # remove old audio
        while len(self.audio_list)>0 and (audio.s-self.audio_list[0].e)>(audio.rate*10):
            self.audio_list.pop(0)

        if len(self.audio_list)==0:
            # first data
            self.audio_list.append(audio)
            return
        if (audio.s-self.audio_list[-1].e)<audio.rate:
            # join audio in 1seconds
            self.audio_list.append(audio)
        else:
            # append audio
            self.audio_list[-1].append(audio)

    def to_messages(self,role:str='user') ->list[dict]:
        res = []
        for audio in self.audio_list:
            res.append( {'role',role, 'content',audio.user_input})
        return res

    def to_content(self) -> str:
        res = []
        for audio in self.audio_list:
            res.append(audio.user_input)
        return "\n\n".join(res)

class TtsAudio:
    """
    Class for handling text-to-speech audio data and its playback.

    This class manages the conversion of text responses to audio, controls playback position,
    and handles the acceptance status of the audio segment.

    Attributes:
        agent_task (AgentTask): Manager for handling AI interactions.
        ai_response (str): The text response to be converted to speech.
        no (int): Sequential number of the response.
        rate (int): The sampling rate of the audio in Hz.
        audio (np.ndarray | None): The audio data, or None if not yet set.
        pos (int): Current playback position in the audio.
        accepted (bool): Whether this audio segment has been accepted/processed.
    """

    def __init__(self, agent_task: 'AgentTask | None', no: int, ai_response: str):
        """
        Initialize a new TTS audio object.

        Args:
            agent_task (AgentTask): Manager for handling AI interactions.
            no (int): Sequential number of the response.
            ai_response (str): The text response to be converted to speech.
        """
        self.agent_task:AgentTask|None = agent_task
        self.ai_response:str = ai_response
        self.no:int = no
        # result of text-to-speech
        self.rate:int = 0
        self.audio:np.ndarray|None = None
        # emit
        self.pos:int = 0
        self.accepted:bool = False

    def set_audio(self, audio:tuple[int, np.ndarray]):
        """
        Set the TTS conversion result.

        Args:
            audio (tuple[int, np.ndarray]): A tuple containing:
                - sampling rate (int)
                - audio data (np.ndarray) as a 1D array, which may be automatically 
                  reshaped to 2D (1, N) format for internal processing
        """
        self.rate = audio[0]
        array = audio[1]
        #if array is not None:
        #     if array.dtype != np.int16 and array.dtype != np.float32 or len(array.shape)!=2 or array[0]!=1:
        #         raise ValueError(f"Audio must be a 1D array of int16 or float32 type {array.dtype} {array.shape}")
        self.audio = array
        self.pos = 0

    def is_canceled(self) -> bool:
        """
        Check if the current TTS processing has been canceled.

        Returns:
            bool: True if canceled, False otherwise.
        """
        return self.agent_task is not None and self.agent_task.is_canceled()

    def is_done(self) -> bool:
        """
        Check if the audio playback has completed.

        Returns:
            bool: True if the audio is None or has finished playing, False otherwise.
        """
        return self.audio is None or len(self.audio)<=self.pos

    def is_accepted(self) -> bool:
        """
        Check and update the acceptance status of the audio segment.
        
        If the audio has played for sufficient duration (>0.8 seconds) or has completed,
        marks it as accepted and registers the response with the AgentTask.

        Returns:
            bool: True if the audio was just accepted, False otherwise.
        """
        if not self.accepted and self.audio is not None:
            if self.pos>=len(self.audio) or (self.pos/self.rate)>0.8:
                self.accepted = True
                if self.agent_task:
                    self.agent_task.accept(self.ai_response)
                return True
        return False

    def get_emit_data(self,duration:float=0.6) -> tuple[int, np.ndarray]|None:
        """
        Extract the next segment of audio data for emission.

        Args:
            duration (float, optional): The duration of the audio segment to extract in seconds.
                Defaults to 0.6 seconds.

        Returns:
            tuple[int, np.ndarray]|None: A tuple containing:
                - sampling rate (int)
                - audio segment (np.ndarray) reshaped to (1, -1), 
                or None if the audio is completed or not yet set.
        """
        if self.audio is None or self.pos>=len(self.audio):
            return None
        step = int(self.rate*duration)
        s = self.pos
        self.pos = min( s+step, len(self.audio))
        segment = self.audio[s:self.pos]
        return (self.rate, segment.reshape(1,-1))

    def get_messages(self) ->list[dict]:
        return self.agent_task.get_messages() if self.agent_task else []

def testtt():
    aa = [ None, "", "aaa" ]
    bb = [ None, "", "bbb" ]
    for a in aa:
        for b in bb:
            xx = str_join(a,b)
            print(f" {a} + {b} = {xx}")

if __name__ == "__main__":
    testtt()
