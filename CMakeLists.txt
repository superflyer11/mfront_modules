#
# MoFEM is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MoFEM is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MoFEM. If not, see <http://www.gnu.org/licenses/>

set(MFRONT_INTERFACE_INSTALL_DIR "mfront_interface")  

include_directories(
  ${UM_SOURCE_DIR}/basic_finite_elements/
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(NOT MGIS_DIR)
  message( WARNING "MGIS_DIR is not defined! MFront Interface will not be configured." )
endif(NOT MGIS_DIR)

set(DEFAULT_LIB_EXTENSION STRING ".so")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(DEFAULT_LIB_EXTENSION STRING ".dylib")
endif()

if(MGIS_DIR)
  # set(MGIS_FOUND TRUE)
  find_library(MGIS_LIBRARY NAMES libMFrontGenericInterface.dylib libMFrontGenericInterface.so PATHS ${MGIS_DIR}/lib)
  find_path(MGIS_HEADER PATHS ${MGIS_DIR}/include/MGIS/)
  add_library(MFrontGenericInterface SHARED IMPORTED)
  set_target_properties(MFrontGenericInterface PROPERTIES IMPORTED_LOCATION ${MGIS_LIBRARY})
  message(STATUS ${MGIS_LIBRARY})
  if(MGIS_LIBRARY)
    # include_directories(${MGIS_HEADER})
    add_definitions(-DWITH_MGIS)
  endif(MGIS_LIBRARY)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
  
add_executable(mfront_interface
    ${CMAKE_CURRENT_SOURCE_DIR}/mfront_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/Operators.cpp
)

include_directories(${MGIS_DIR}/include)
include_directories(${MGIS_DIR}/lib)

add_library(mfront_lib_all
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/Operators.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/MFrontMoFEMInterface.cpp
)

install(
  TARGETS mfront_lib_all
  DESTINATION ${MFRONT_INTERFACE_INSTALL_DIR}
  EXPORT mfront_lib_all_targets)
install(
  EXPORT mfront_lib_all_targets
  FILE mfront_lib_all_targets.cmake
  DESTINATION ${MFRONT_INTERFACE_INSTALL_DIR})

add_dependencies(mfront_interface MFrontGenericInterface)

target_link_libraries(mfront_interface
    MFrontGenericInterface
    mfront_lib_all
    users_modules
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_third_party
    ${MoFEM_PROJECT_LIBS}
  )

install(TARGETS mfront_interface DESTINATION ${MFRONT_INTERFACE_INSTALL_DIR})

function(mp_copy_file target)
file(
  COPY ${target}
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)
endfunction(mp_copy_file)

mp_copy_file(${CMAKE_CURRENT_SOURCE_DIR}/compile_behaviours.sh)
mp_copy_file(${CMAKE_CURRENT_SOURCE_DIR}/README.md)
mp_copy_file(${CMAKE_CURRENT_SOURCE_DIR}/load_history.in)

add_custom_target(
  mfront_make_link_to_behaviours_directory
  ALL
  COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/behaviours ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Make linking to behaviours directory" VERBATIM)

endif(MGIS_DIR)

if(TFEL_DIR)
  message("TFEL DIR FOUND")
  add_test(compile_behaviours_test
    ${TFEL_DIR}/bin/mfront --obuild --interface=generic ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/Plasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogPlastNeck.mfront)
endif(TFEL_DIR)

get_property(test_names DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
set_property(TEST ${test_names} PROPERTY LABELS "mfront-interface-tests")
# set_property(TEST ${test_names} PROPERTY TIMEOUT 60)
