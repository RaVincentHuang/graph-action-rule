
class DatasetConfig:
    def __init__(self, sampler: str, total_num: int, node_num: int, node_random=False, node_feature="node"):
        self.sampler = sampler
        self.total_num = total_num
        self.node_num = node_num
        self.node_feature = node_feature