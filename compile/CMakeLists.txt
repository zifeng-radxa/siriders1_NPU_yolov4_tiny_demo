cmake_minimum_required(VERSION 3.6.0)

set(AIPU_VERSION r1p3)

#set(UMDSRC "${CMAKE_SOURCE_DIR}/../AI610-SDK-${AIPU_VERSION}-00eac0/AI610-SDK-1012-${AIPU_VERSION}-eac0/Linux-driver/driver/umd")
set(UMDSRC "${CMAKE_SOURCE_DIR}/../../AI610-SDK-${AIPU_VERSION}-AIoT/AI610-SDK-r1p3-00eac0/AI610-SDK-1012-${AIPU_VERSION}-eac0/Linux-driver/driver/umd")

if(ANDROID)
  message("android")
  set(ANDROID_NDK_ROOT "/opt/android-ndk-r23c/toolchains/llvm/prebuilt/linux-x86_64/bin")
  set(CMAKE_CXX_COMPILER ${ANDROID_NDK_ROOT}/aarch64-linux-android31-clang++)
  set(CMAKE_C_COMPILER ${ANDROID_NDK_ROOT}/aarch64-linux-android31-clang)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/android/libs)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/android)
else()
  message("linux")
  set(Linux_Tool_ROOT "/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin")
  set(CMAKE_CXX_COMPILER ${Linux_Tool_ROOT}/aarch64-linux-gnu-g++)
  set(CMAKE_C_COMPILER ${Linux_Tool_ROOT}/aarch64-linux-gnu-gcc)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/linux/libs)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/linux)
endif()

add_subdirectory(./umd)
add_subdirectory(./aiputest)
