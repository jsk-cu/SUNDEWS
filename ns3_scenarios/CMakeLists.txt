# CMakeLists.txt for SatUpdate NS-3 Scenario
#
# This file configures the build for the satellite-update-scenario.
# It should be placed in the ns-3 scratch directory or configured
# as a custom module.
#
# Usage:
#   1. Copy this directory to ${NS3_DIR}/scratch/satellite-update-scenario/
#   2. Run: ./ns3 build
#   3. Execute: ./ns3 run satellite-update-scenario
#
# Or configure as an external project linking to NS-3 libraries.

cmake_minimum_required(VERSION 3.10)
project(satellite-update-scenario VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find NS-3 installation
# Set NS3_DIR to your NS-3 installation path
if(NOT DEFINED NS3_DIR)
    set(NS3_DIR "/usr/local/ns3" CACHE PATH "Path to NS-3 installation")
endif()

# NS-3 library paths
set(NS3_BUILD_DIR "${NS3_DIR}/build")
set(NS3_INCLUDE_DIR "${NS3_DIR}/build")

# Check if NS-3 is available
if(NOT EXISTS "${NS3_DIR}/ns3")
    message(WARNING "NS-3 not found at ${NS3_DIR}. Please set NS3_DIR correctly.")
endif()

# Include directories
include_directories(
    ${NS3_INCLUDE_DIR}
    ${NS3_DIR}/src
)

# Link directories
link_directories(
    ${NS3_BUILD_DIR}/lib
)

# Source files
set(SOURCES
    satellite-update-scenario.cc
)

# Create executable
add_executable(satellite-update-scenario ${SOURCES})

# NS-3 modules to link
# These are the core modules needed for satellite simulation
set(NS3_MODULES
    ns3-core
    ns3-network
    ns3-internet
    ns3-point-to-point
    ns3-applications
    ns3-mobility
)

# Link NS-3 libraries
target_link_libraries(satellite-update-scenario
    ${NS3_MODULES}
)

# Installation
install(TARGETS satellite-update-scenario
    RUNTIME DESTINATION bin
)

# Configuration file template
configure_file(
    ${CMAKE_SOURCE_DIR}/config.json.in
    ${CMAKE_BINARY_DIR}/config.json
    @ONLY
)

# Custom target for running with test input
add_custom_target(run-test
    COMMAND satellite-update-scenario --input=${CMAKE_SOURCE_DIR}/test_input.json --output=${CMAKE_BINARY_DIR}/test_output.json --verbose
    DEPENDS satellite-update-scenario
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running satellite-update-scenario with test input"
)

# Print configuration
message(STATUS "NS-3 Directory: ${NS3_DIR}")
message(STATUS "NS-3 Build Directory: ${NS3_BUILD_DIR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")