import ctypes


class DatasetConfig:
    def __init__(self, sampler: str, total_num: int, node_num: int, node_num_random=lambda x:x, node_feature="node"):
        self.sampler = sampler
        self.total_num = total_num
        self.node_num = node_num
        self.node_feature = node_feature
        self.node_num_random = node_num_random
        
    def __format__(self, format_spec: str) -> str:
        return f"sampler{self.sampler}-total_num{self.total_num}-node_num{self.node_num}-node_feature{self.node_feature}"

class SubgraphMatchingConfig:
    def __init__(self, filter_type, order_type, engine_type, order_num, time_limit) -> None:
        self.filter_type = filter_type
        self.order_type = order_type
        self.engine_type = engine_type
        self.order_num = order_num
        self.time_limit = time_limit
    
    def __format__(self, format_spec: str) -> str:
        return f"filter_type{self.filter_type}-order_type{self.order_type}-engine_type{self.engine_type}-order_num{self.order_num}-time_limit{self.time_limit}"
    
    # char* filter_type, char* order_type, char* engine_type, int order_num, int time_limit
    def get_ctype(self) -> tuple[ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]:
        return (ctypes.c_char_p(self.filter_type.encode('utf-8')),
            ctypes.c_char_p(self.order_type.encode('utf-8')),
            ctypes.c_char_p(self.engine_type.encode('utf-8')),
            ctypes.c_int(self.order_num),
            ctypes.c_int(self.time_limit))
        

class SearchConfig:
    def __init__(self, method_generate, method_evaluate, method_select, n_select_sample, n_evaluate_sample, n_generate_sample, prompt_sample, stop, cache_value=True):
        self.method_generate = method_generate
        self.method_evaluate = method_evaluate
        self.method_select = method_select
        self.n_select_sample = n_select_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_generate_sample = n_generate_sample
        self.prompt_sample = prompt_sample
        self.stop = stop
        self.cache_value = cache_value
        
class LLMConfig:
    def __init__(self, model, temperature, max_tokens, n, stop) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop