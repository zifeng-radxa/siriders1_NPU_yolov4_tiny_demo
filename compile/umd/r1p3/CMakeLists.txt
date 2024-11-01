set(TARGET_NAME aipudrv)

set(AIPU_MAJOR 1)
set(AIPU_MINOR 3)
set(AIPU_PATCH 0)

option(ARM_LINUX "ARM_LINUX" ON)
add_definitions(-DARM_LINUX=1)
add_definitions(-DZHOUYI_V12)
add_definitions(-DRTDEBUG=0)
add_definitions(-DMACRO_UMD_VERSION="5.5.0")

set(INCLUDE_DIRS "")
list(APPEND INCLUDE_DIRS ${UMDSRC}/src)
list(APPEND INCLUDE_DIRS ${UMDSRC}/src/common)
list(APPEND INCLUDE_DIRS ${UMDSRC}/src/device)
list(APPEND INCLUDE_DIRS ${UMDSRC}/src/zhouyi_v1v2)
list(APPEND INCLUDE_DIRS ${UMDSRC}/src/zhouyi_v3)
list(APPEND INCLUDE_DIRS ${UMDSRC}/src/misc)
list(APPEND INCLUDE_DIRS ${UMDSRC}/src/utils)
list(APPEND INCLUDE_DIRS ${UMDSRC}/include)

set(AIPU_SRC_FILES "")
list(APPEND AIPU_SRC_FILES ${UMDSRC}/src/common/context.cpp
        ${UMDSRC}/src/common/ctx_ref_map.cpp
        ${UMDSRC}/src/common/graph.cpp
        ${UMDSRC}/src/common/graph_base.cpp
        ${UMDSRC}/src/common/job_base.cpp
        ${UMDSRC}/src/common/parser_base.cpp
        ${UMDSRC}/src/common/memory_base.cpp
        ${UMDSRC}/src/common/standard_api_impl.cpp
        ${UMDSRC}/src/common/status_string.cpp
        ${UMDSRC}/src/misc/aipu_printf.cpp
        ${UMDSRC}/src/utils/helper.cpp
        ${UMDSRC}/src/device/aipu/aipu.cpp
        ${UMDSRC}/src/device/aipu/ukmemory.cpp
        ${UMDSRC}/src/zhouyi_v1v2/graph_v1v2.cpp
        ${UMDSRC}/src/zhouyi_v1v2/job_v1v2.cpp
        ${UMDSRC}/src/zhouyi_v1v2/parser_v1v2.cpp)

include_directories(${INCLUDE_DIRS})

add_library(${TARGET_NAME}  SHARED  ${AIPU_SRC_FILES})

SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE C)
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${AIPU_MAJOR}.${AIPU_MINOR}.${AIPU_PATCH})
