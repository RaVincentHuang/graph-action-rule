file(GLOB_RECURSE GRAPH_SRC "./*.cpp")
add_library(graph SHARED ${GRAPH_SRC})
target_include_directories(graph PUBLIC ${PROJECT_SOURCE_DIR}/include/graph ${PROJECT_SOURCE_DIR}/include)
