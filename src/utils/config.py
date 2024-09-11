
class DatasetConfig:
    def __init__(self, sampler: str, total_num: int, node_num: int, node_num_random=lambda x:x, node_feature="node"):
        self.sampler = sampler
        self.total_num = total_num
        self.node_num = node_num
        self.node_feature = node_feature
        self.node_num_random = node_num_random
        
    def __format__(self, format_spec: str) -> str:
        return f"sampler{self.sampler}-total_num{self.total_num}-node_num{self.node_num}-node_feature{self.node_feature}"