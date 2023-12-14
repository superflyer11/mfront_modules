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
  # find_path(MGIS_HEADER PATHS ${MGIS_DIR}/include/MGIS/)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/MFrontOperators.cpp
)

include_directories(${MGIS_DIR}/include)
include_directories(${MGIS_DIR}/lib)

add_library(mfront_lib_all
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/MFrontOperators.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/MFrontMoFEMInterface.cpp
)

target_link_libraries(mfront_lib_all 
  MFrontGenericInterface
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

target_link_libraries(mfront_interface
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
  COMMENT "Make linking to behaviours directory" VERBATIM
)
  
add_custom_target(mfront_compile_behaviours
    ${TFEL_DIR}/bin/mfront --obuild --interface=generic ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogPlastNonlinHard.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogPlastNonlinHardNoParam.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/IsotropicLinearHardeningPlasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/StandardLinearSolid.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LinearElasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/SignoriniHyperElasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/OgdenHyperElasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/SaintVenantKirchhoffElasticity.mfront
)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.so)
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.so DESTINATION ${MFRONT_INTERFACE_INSTALL_DIR} 
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
    GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.so)
    add_dependencies(mfront_make_link_to_behaviours_directory mfront_compile_behaviours)
  endif()

endif(MGIS_DIR)

if(TFEL_DIR)
  message("TFEL_DIR FOUND")

  ##############################
  ##### COMPILE BEHAVIOURS #####
  ##############################

  add_test(compile_behaviours_test
    ${TFEL_DIR}/bin/mfront --obuild --interface=generic ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogPlastNonlinHard.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LogPlastNonlinHardNoParam.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/StandardLinearSolid.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/LinearElasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/SignoriniHyperElasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/OgdenHyperElasticity.mfront ${CMAKE_CURRENT_SOURCE_DIR}/behaviours/SaintVenantKirchhoffElasticity.mfront)

  ####################
  ##### 3D TESTS #####
  ####################

  ### ELASTOPLASTICITY 3D ###

  add_test(3D_necking_plate
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/3D_necking_coarse_4parts.h5m -mi_block_1 LogPlastNonlinHardNoParam -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -ts_max_time 1.0 -ts_dt 0.02 -snes_atol 1e-8 -snes_rtol 1e-8 -snes_max_it 65  -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_theta_initial_guess_extrapolate 1 -ts_theta_theta 1  -ts_exact_final_time matchstep -atom_test 1 -field_eval_coords 6.2975,0,0 -mi_test_jacobian 1 -mi_random_field_scale 0.001 -is_partitioned 1 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history.in
  )

  add_test(necking_miehe
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/necking_miehe_fine_4parts.h5m -mi_block_1 LogPlastNonlinHard -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 450 -mi_param_1_1 129.24 -mi_param_1_2 265 -mi_param_1_3 16.93 -ts_max_time 1.4 -ts_dt 0.02 -snes_atol 1e-8 -snes_rtol 1e-8 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.001 -atom_test 2 -field_eval_coords 6.298,0,0 -is_partitioned 1 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history.in
  )

  ### VISCOELASTICITY 3D ###

  add_test(relaxation_sls_plate
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/relax_plate_4parts.h5m -mi_block_1 StandardLinearSolid -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 23333.33333333 -mi_param_1_1 35000.0 -mi_param_1_2 6666.6666 -mi_param_1_3 10000 -mi_param_1_4 0.05 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history/creep_test.in -ts_max_time 0.5 -ts_dt 0.005 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_save_gauss -atom_test 3 -is_partitioned 1 -mi_test_jacobian 1 -mi_random_field_scale 0.1 -snes_lag_jacobian -2
  )

  add_test(creep_sls_plate
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/creep_plate_4parts.h5m -mi_block_1 StandardLinearSolid -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 23333.33333333 -mi_param_1_1 35000.0 -mi_param_1_2 6666.6666 -mi_param_1_3 10000 -mi_param_1_4 0.05 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history/creep_test.in -ts_max_time 0.5 -ts_dt 0.005 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_save_gauss -atom_test 4 -is_partitioned 1 -mi_test_jacobian 1 -mi_random_field_scale 0.1 -snes_lag_jacobian -2
  )

  add_test(recovery_sls_plate
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/creep_plate_4parts.h5m -mi_block_1 StandardLinearSolid -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 23333.33333333 -mi_param_1_1 35000.0 -mi_param_1_2 6666.6666 -mi_param_1_3 10000 -mi_param_1_4 0.05 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history/recovery_test.in -ts_max_time 1.0 -ts_dt 0.005 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_save_gauss -atom_test 5 -is_partitioned 1 -mi_test_jacobian 1 -mi_random_field_scale 0.1 -snes_lag_jacobian -2
  )

  ### ELASTICITY 3D ###

  add_test(cook_membrane_linear_elastic_3d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_4parts.h5m -mi_block_1 LinearElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 240.565 -mi_param_1_1 0.4999 -mi_save_gauss -ts_max_time 1.0 -ts_dt 1.0 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 1 -is_partitioned 1 -atom_test 6 -field_eval_coords 48,60,0 -mi_save_volume 1 -mi_save_gauss 1
  )

  add_test(lame_cylinder_linear_elastic_3d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/lame_cylinder_3d_4parts.h5m -mi_block_1 LinearElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 500 -mi_param_1_1 0.35 -mi_save_gauss -ts_max_time 1.0 -ts_dt 1.0 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 1 -is_partitioned 1 -atom_test 8 -field_eval_coords 1,0,0 -mi_save_volume 1 -mi_save_gauss 1
  )

  ### HYPERELASTICITY 3D ###

  add_test(cook_membrane_neohookean_3d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_4parts.h5m -mi_block_1 SignoriniHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 40.0942e4 -mi_param_1_1 40.0969 -mi_param_1_2 0 -mi_param_1_3 0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 0.2 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 7 -field_eval_coords 48,60,0
  )

  add_test(cook_membrane_mooney_rivlin_3d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_4parts.h5m -mi_block_1 SignoriniHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 40.0942e4 -mi_param_1_1 20.04845 -mi_param_1_2 20.04845 -mi_param_1_3 0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 0.05 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 7 -field_eval_coords 48,60,0
  )

  add_test(cook_membrane_ogden_3d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_4parts.h5m -mi_block_1 OgdenHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 40.0942e4 -mi_param_1_1 80.1938 -mi_param_1_2 2.0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 0.05 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 7 -field_eval_coords 48,60,0
  )

  # add_test(cook_membrane_linear_elastic_compress_3d
  #   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface {CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_compress.cub -mi_block_1 LinearElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 500 -mi_param_1_1 0.35 -mi_save_gauss -ts_max_time 1.0 -ts_dt 1.0 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 1 -is_partitioned 1 -atom_test 8 -field_eval_coords 48,60,0 -mi_save_volume 1 -mi_save_gauss 1
  # )

  # add_test(cook_membrane_neohookean_compress_3d
  #   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface {CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_compress.cub -mi_block_1 SignoriniHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 555.5556 -mi_param_1_1 92.5925 -mi_param_1_2 0 -mi_param_1_3 0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 1.0 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 8 -field_eval_coords 48,60,0 -mi_save_volume 1 -mi_save_gauss 1
  # )

  
  ####################
  ##### 2D TESTS #####
  ####################

  ### ELASTOPLASTICITY 2D ###

  add_test(necking_2d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/necking_2d_4parts.h5m -hypothesis plane_strain -mi_block_1 LogPlastNonlinHardNoParam -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -ts_max_time 1.0 -ts_dt 0.01 -snes_atol 1e-8 -snes_rtol 1e-8 -snes_max_it 65  -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_theta_initial_guess_extrapolate 1 -ts_theta_theta 1  -ts_exact_final_time matchstep -atom_test 1 -field_eval_coords 6.2975,0 -mi_test_jacobian 1 -mi_random_field_scale 0.01 -is_partitioned 1 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history.in
)

  add_test(necking_miehe_2d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/necking_2d_4parts.h5m -hypothesis axisymmetry -mi_block_1 LogPlastNonlinHard -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} 
    -mi_param_1_0 450 -mi_param_1_1 129.24 -mi_param_1_2 265 -mi_param_1_3 16.93 -ts_max_time 1.4 -ts_dt 0.02 -snes_atol 1e-8 -snes_rtol 1e-8 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type bt -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.01 -atom_test 2 -field_eval_coords 6.2975,0 -is_partitioned 1 -load_history ${CMAKE_CURRENT_SOURCE_DIR}/load_history.in
  )

  ### ELASTICITY 2D ###

  add_test(cook_membrane_linear_elastic_2d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_2d_4parts.h5m -hypothesis plane_strain -mi_block_1 LinearElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 240.565 -mi_param_1_1 0.4999 -mi_save_gauss -ts_max_time 1.0 -ts_dt 1.0 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 1 -is_partitioned 1 -atom_test 6 -field_eval_coords 48,60 -mi_save_volume 1
  )


  add_test(lame_cylinder_linear_elastic_axisymm
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/lame_cylinder_axi_4parts.h5m -hypothesis axisymmetry -mi_block_1 LinearElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 500 -mi_param_1_1 0.35 -ts_max_time 1.0 -ts_dt 1.0 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 1 -is_partitioned 1 -field_eval_coords 1,0,0 -mi_save_volume 1 -mi_save_gauss 1 -atom_test 8
  )

  ### HYPERELASTICITY 2D ###

  add_test(cook_membrane_neohookean_2d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_2d_4parts.h5m -hypothesis plane_strain -mi_block_1 SignoriniHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 40.0942e4 -mi_param_1_1 40.0969 -mi_param_1_2 0 -mi_param_1_3 0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 0.05 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 7 -field_eval_coords 48,60 -mi_save_volume 1
  )

  add_test(cook_membrane_mooney_rivlin_2d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_2d_4parts.h5m -hypothesis plane_strain -mi_block_1 SignoriniHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 40.0942e4 -mi_param_1_1 20.04845 -mi_param_1_2 20.04845 -mi_param_1_3 0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 0.05 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 7 -field_eval_coords 48,60,0
  )

  add_test(cook_membrane_ogden_2d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/mfront_interface -file_name ${CMAKE_CURRENT_SOURCE_DIR}/meshes/cook_membrane_2d_4parts.h5m -hypothesis plane_strain -mi_block_1 OgdenHyperElasticity -mi_lib_path_1 ${CMAKE_CURRENT_BINARY_DIR}/src/libBehaviour.${DEFAULT_LIB_EXTENSION} -mi_param_1_0 40.0942e4 -mi_param_1_1 80.1938 -mi_param_1_2 2.0 -mi_save_gauss -ts_max_time 1.0 -ts_dt 0.05 -snes_atol 1e-12 -snes_rtol 1e-12 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -snes_linesearch_type basic -ts_type beuler -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep -pc_type lu -mi_test_jacobian 1 -mi_random_field_scale 0.1 -is_partitioned 1 -atom_test 7 -field_eval_coords 48,60,0
  )

endif(TFEL_DIR)

if(NOT TFEL_DIR)
  message( WARNING "TFEL_DIR is not defined! Unable to test behavior compilation." )
endif(NOT TFEL_DIR)

get_property(test_names DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
set_property(TEST ${test_names} PROPERTY LABELS "mfront-interface-tests")
# set_property(TEST ${test_names} PROPERTY TIMEOUT 60)

