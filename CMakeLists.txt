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

include_directories(
  ${PROJECT_SOURCE_DIR}/basic_finite_elements/nonlinear_elastic_materials/src)

if(NOT MGIS_DIR)
  message( WARNING "MGIS_DIR is not defined! MFront Interface will not be configured." )
endif(NOT MGIS_DIR)

if(MGIS_DIR)
  message("MGIS_DIR FOUND")
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

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_compile_definitions(mfront_lib_all PUBLIC DEFAULT_LIB_EXTENSION="dylib")
    set(DEFAULT_LIB_EXTENSION "dylib")
  else()
    target_compile_definitions(mfront_lib_all PUBLIC DEFAULT_LIB_EXTENSION="so")
    set(DEFAULT_LIB_EXTENSION "so")
endif()

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
    mofem_matrix_function
    mofem_post_proc
    mofem_boundary_conditions
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
  message("TFEL_DIR FOUND")

  add_test(compile_behaviours_test
    ${TFEL_DIR}/bin/mfront --obuild --interface=generic ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/Plasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogPlastNeck.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogarithmicStrainPlasticityMiehe.mfront)

endif(TFEL_DIR)

if(NOT TFEL_DIR)
  message( WARNING "TFEL_DIR is not defined! Unable to test behavior compilation." )
endif(NOT TFEL_DIR)

# add_test(mi_necking_miehe_test
# ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface
# -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/necking_miehe_fine.cub
# -mi_block_1 LogarithmicStrainPlasticityMiehe -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_print_gauss 1 -order 2 -mi_param_1_0 450 -mi_param_1_1 129 -mi_param_1_2 265 -mi_param_1_3 16.93 -ts_dt 0.005 -ksp_monitor -log_sl verbose -pc_type lu -ts_type theta -ts_adapt_always_accept 1 -ts_theta_initial_guess_extrapolate 0 -ts_theta_theta 1 -ts_max_time 0.2 -ts_adapt_type none -field_eval_coords 0,-6.298,0 -mi_test_jacobian 0 -mi_random_field_scale 0.1 -is_partitioned 0
# )

add_test(mi_necking_miehe_test
${CMAKE_CURRENT_BINARY_DIR}/mfront_interface
-file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/3D_necking_coarse.cub
-mi_block_1 LogarithmicStrainPlasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -ts_max_time 1.0 -ts_dt 0.01 -order 2 -snes_atol 1e-7 -snes_rtol 1e-7 -snes_max_it 40 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_type theta -ts_theta_initial_guess_extrapolate 1 -ts_theta_theta 1  -ts_exact_final_time matchstep -is_partitioned 0 -mi_print_gauss 1
)

# add_test(mi_necking_miehe_test
# ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface
# -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/3D_necking_coarse.cub
# -mi_block_1 LogarithmicStrainPlasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -order 2 -ts_dt 0.01 -ts_type theta -ts_adapt_always_accept 1 -ts_theta_initial_guess_extrapolate 0 -ts_theta_theta 1 -ts_max_time 0.2 -ts_adapt_type none -field_eval_coords 0,-6.298,0 -is_partitioned 0
# )

get_property(test_names DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
set_property(TEST ${test_names} PROPERTY LABELS "mfront-interface-tests")
# set_property(TEST ${test_names} PROPERTY TIMEOUT 60)

