cmake_minimum_required(VERSION 3.8)
project(rpc)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/rpc_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(rpc_dynamics_tests test)
endif()

ament_package()

