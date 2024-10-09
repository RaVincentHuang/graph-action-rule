file(GLOB_RECURSE UTILITY_SRC "./*.cpp")
file(GLOB_RECURSE HAN_SRC "han/*.cpp")
add_library(utility SHARED ${UTILITY_SRC} ${HAN_SRC})
target_include_directories(utility PUBLIC ${PROJECT_SOURCE_DIR}/include/utility ${PROJECT_SOURCE_DIR}/include)
