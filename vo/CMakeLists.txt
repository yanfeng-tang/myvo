cmake_minimum_required(VERSION 2.8.3)
project(myvo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
	roscpp
	std_msgs
	geometry_msgs
	tf
	cv_bridge
	camera_model
)

find_package(OpenCV REQUIRED)
catkin_package()

find_package(Ceres REQUIRED)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3  REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(myvo
	src/factor/optimization.cpp
	#src/factor/visualization.cpp
	src/factor/projection_factor.cpp
	src/parameter.cpp
	src/rosNode.cpp
	src/estimator.cpp
	src/feature_manager.cpp
)

target_link_libraries(myvo ${catkin_LIBRARIES} ${CERES_LIBRARIES})
