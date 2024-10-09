set(SUBGRAPH_MATCHING_FILE_LIST
        matchingcommand.cpp
        FilterVertices.cpp
        BuildTable.cpp
        GenerateQueryPlan.cpp
        EvaluateQuery.cpp
        GenerateFilteringPlan.cpp
        # StudyPerformance.cpp
        Match.cpp)
        
add_library(matching SHARED ${SUBGRAPH_MATCHING_FILE_LIST})
target_link_libraries(matching graph utility)
target_include_directories(matching PUBLIC ${PROJECT_SOURCE_DIR}/include/matching ${PROJECT_SOURCE_DIR}/include)

# set(GRAPH_CONVERTER_FILE_LIST
#                 GraphConverter.cpp)
# add_executable(GraphConverter.out ${GRAPH_CONVERTER_FILE_LIST})
# target_link_libraries(GraphConverter.out graph utility)

# add_executable(EdgeListConverter.out EdgeListToCSR.cpp)
# add_executable(ReassignLabel ReassignLabel.cpp)
# add_executable(SelectEdge SelectEdge.cpp)
