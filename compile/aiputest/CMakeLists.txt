
list(APPEND INCLUDE_DIRS ${UMDSRC}/include)

include_directories(${INCLUDE_DIRS})

add_executable( aipu_test  main.cpp)
if (DEFINED ANDROID)
   target_link_libraries( aipu_test aipudrv -pthread)
else()
   target_link_libraries( aipu_test aipudrv -lpthread)
endif()

