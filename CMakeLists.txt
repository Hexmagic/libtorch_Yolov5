cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(app)
set(CMAKE_PREFIX_PATH "libtorch_example/libtorch")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(Torch_DIR "libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( 
    ${OpenCV_INCLUDE_DIRS}    
)
add_executable(main main.cpp )
set(CMAKE_INSTALL_PREFIX "..")
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})



