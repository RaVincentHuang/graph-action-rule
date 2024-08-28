from littleballoffur import MetropolisHastingsRandomWalkSampler, RandomWalkSampler, FrontierSampler, CommunityStructureExpansionSampler, \
    NonBackTrackingRandomWalkSampler, DepthFirstSearchSampler, BreadthFirstSearchSampler, DiffusionSampler, DiffusionTreeSampler, ForestFireSampler, \
    SpikyBallSampler, SnowBallSampler, CirculatedNeighborsRandomWalkSampler
import networkx as nx
def select_sampler(sampler_name: str):
    match sampler_name:
        case "metropolis_hastings_random_walk":
            return MetropolisHastingsRandomWalkSampler
        case "random_walk":
            return RandomWalkSampler
        case "frontier":
            return FrontierSampler
        case "community_structure_expansion":
            return CommunityStructureExpansionSampler
        case "non_back_tracking_random_walk":
            return NonBackTrackingRandomWalkSampler
        case "dfs":
            return DepthFirstSearchSampler
        case "bfs":
            return BreadthFirstSearchSampler
        case "diffusion":
            return DiffusionSampler
        case "diffusion_tree":
            return DiffusionTreeSampler
        case "forest_fire":
            return ForestFireSampler
        case "spiky_ball":
            return SpikyBallSampler
        case "snow_ballSampler":
            return SnowBallSampler
        case "circulated_neighbors_random_walk":
            return CirculatedNeighborsRandomWalkSampler
        case _:
            raise ValueError(f"Sampler {sampler_name} not found.")

# TODO 考虑更好的方法
def sample_graph(graph: nx.DiGraph, sampler: str, node_num: int) -> nx.DiGraph:
    undir_graph = nx.Graph(graph)
    model = select_sampler(sampler)(number_of_nodes=node_num)
    undir_subgraph = model.sample(undir_graph)
    subgraph = graph.subgraph(undir_subgraph.nodes)
    return subgraph

