from abc import ABC, abstractmethod

DATA_PATH = "/home/vincent/graphrule/data/tasks/"

class Task(ABC):
    @abstractmethod
    def __init__(self):
        self.data = []
        self.value_cache = {}
        self.steps = 0
        self.stops = []
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_input(self, idx: int) -> str:
        pass

    @abstractmethod
    def test_output(self, idx: int, output: str) -> dict[str, int]:
        pass
    
    @staticmethod
    @abstractmethod
    def standard_prompt_wrap(x: str, y: str) -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def cot_prompt_wrap(x: str, y: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        pass
    
    @staticmethod
    @abstractmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        pass
