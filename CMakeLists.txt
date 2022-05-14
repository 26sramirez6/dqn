cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lux_cpp)

find_package(Torch REQUIRED)


set(CMAKE_INCLUDE_CURRENT_DIR ON)

# newer versions of gcc will report redundant warnings while building eigen
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
	add_compile_options(-Wno-deprecated -Wno-ignored-attributes -Wno-narrowing)
endif()
SET(GCC_COVERAGE_LINK_FLAGS    "-lrt")

add_executable(main main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
target_link_libraries(main ${TORCH_LIBRARIES})
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR})

add_subdirectory(googletest)
#add_subdirectory(UnitTest)

