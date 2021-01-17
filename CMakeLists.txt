cmake_minimum_required(VERSION 3.0.0)
project(capsbasic VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(capsbasic capsbasic.cpp)

set(OpenGL_INCLUDE_DIRS "C:/Program\ Files\ (x86)/IntelSWTools/system_studio_2020/OpenCL/sdk/include/")
include_directories(./ ${OpenGL_INCLUDE_DIRS})

set(OpenGL_LID_DIR "C:/Program\ Files\ (x86)/IntelSWTools/system_studio_2020/OpenCL/sdk/lib/x64")
#SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
#set(CMAKE_MODULE_PATH  ${OpenGL_LID_DIR})
#set(CMAKE_PREFIX_PATH ${OpenGL_LID_DIR})
#link_directories(${OpenGL_LID_DIR})

#find_library(OpenCL_LIBS
#    NAMES OpenCL.lib
#    PATHS ${OpenGL_LID_DIR}
#    NO_DEFAULT_PATH)

file(GLOB test_LIBS "${OpenGL_LID_DIR}/*.lib")

target_link_libraries(capsbasic ${test_LIBS})

#message("LOGD ${OpenCL_LIBS}")
message("LOGD ${test_LIBS}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
