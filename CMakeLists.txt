cmake_minimum_required(VERSION 3.0 )
project(app)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(Torch_DIR "libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(main main.cpp  src/detector.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})



