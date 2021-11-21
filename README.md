MFront Interface module   {#um_mfront_interface_readnme}
========================

This user module provides a generic interface for using material behaviours in [MFront library](http://tfel.sourceforge.net/news.html).


Install MFront Interface
=========================
In order to use MFront Interface in MoFEM, first the *mgis* library needs to be installed on the system. 
The most straightforward method is to use [Spack](https://spack.readthedocs.io/en/latest/). 

```
spack install mgis
```

In order to make sure that the installation of MoFEM and *mgis* proceeds without any conflicts it is recommented to use [this branch](https://github.com/likask/spack/tree/develop_upstream_master) of the Spack and the following command:

```
spack install mofem-cephas@develop+mgis
```

For a more detailed instructions on installing MoFEM follow [this link](http://mofem.eng.gla.ac.uk/mofem/html/installation.html). 
To make sure that *mgis* library is properly loaded it is recommended to execute the following line:

```
spack load mgis tfel
```

or add it to *.bashrc* or *.bash_profile* files.

To install MFront Interface module clone this repository into MoFEM users modules source directory, for example:

```
cd $MOFEM_INSTALL_DIR/mofem-cephas/mofem/users_modules/users_modules/
git clone https://karol41@bitbucket.org/karol41/um_mfront_interface.git mfront_interface
```

<!-- Next, the build of the users modules needs to be reconfigured:

```
cd $MOFEM_INSTALL_DIR/users_modules_build
export MGIS_PATH=$(spack find -l --path mgis | awk 'END{print}' | awk 'NF{ print $NF }')"/lib"
touch CMakeCache.txt
./spconfig -DMGIS_PATH=$MGIS_PATH
make –j4
```

Material behaviours
=========================
To compile particular MFront material behaviours, use the provided script:
```
./compile_behaviours.sh ./behaviours/ImplicitNorton.mfront
```

<<<<<<< HEAD
cd bone_remodelling
make
 -->
=======
Example command lines:
(work in progress)

>>>>>>> 05dbb635e0915b9aae0278006c61f9ac84bf5604

#test for beam

```
../tools/mofem_part -my_file ~/Desktop/Meshes/MFront_testing/beam_fenics.cub  -my_nparts 10 &&  mpirun -np 10 /home/karol/mofem_install/users_modules/mfront_interface/mfront_interface -file_name out.h5m -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps -ts_max_snes_failures 1 -block_1 LogarithmicStrainPlasticity  -param_1_0 250e6 -param_1_1 1e6 -snes_rtol 1e-7 -snes_atol 1e-7 -ksp_rtol 1e-12 -ksp_atol 1e-12 -ts_adapt_type none -print_gauss -load_history load_history.in -order 2  -ts_dt 0.05 -ts_adapt_dt_max 0.05  -ts_max_time 1 -snes_monitor -ts_monitor -log_quiet -ts_type theta -ts_adapt_always_accept 1 -ts_theta_initial_guess_extrapolate 1  -ts_theta_theta 1 -ts_adapt_reject_safety 0.9 -ts_max_snes_failures 10  -snes_max_it 20 -ts_exact_final_time matchstep
```

#miehe necking 

```
../tools/mofem_part -my_file /home/karol/Desktop/Meshes/MFront_testing/3D_miehe_necking.cub -my_nparts 10 &&  mpirun -np 10 /home/karol/mofem_install/users_modules/mfront_interface/mfront_interface -file_name out.h5m -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps -ts_max_snes_failures 1 -block_1 LogarithmicStrainPlasticity -param_1_0 450 -param_1_1 130 -snes_rtol 1e-7 -snes_atol 1e-7 -ksp_rtol 1e-12 -ksp_atol 1e-12 -ts_adapt_type basic -print_gauss -load_history load_history.in -order 2  -ts_dt 0.005 -ts_adapt_dt_max 0.02  -ts_max_time 2.5 -snes_monitor -ts_monitor -log_quiet -ts_type theta -ts_adapt_always_accept 1 -ts_theta_initial_guess_extrapolate 1  -ts_theta_theta 1 -ts_adapt_reject_safety 0.9 -ts_max_snes_failures 10  -snes_max_it 20 -ts_exact_final_time matchstep
```


#validation with mofem

```
../tools/mofem_part -my_file /home/karol/Desktop/Meshes/MFront_testing/3D_necking_coarse.cub -my_nparts 10 &&  mpirun -np 10 /home/karol/mofem_install/users_modules/mfront_interface/mfront_interface -file_name out.h5m -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps -ts_max_snes_failures 1 -block_1 LogarithmicStrainPlasticity -param_1_0 450 -param_1_1 10000 -snes_rtol 1e-7 -snes_atol 1e-7 -ksp_rtol 1e-12 -ksp_atol 1e-12 -ts_adapt_type none -print_gauss -load_history load_history.in -order 2  -ts_dt 0.01 -ts_adapt_dt_max 0.02  -ts_max_time 1 -snes_monitor -ts_monitor -log_quiet -ts_type theta -ts_adapt_always_accept 1 -ts_theta_initial_guess_extrapolate 1  -ts_theta_theta 1 -ts_adapt_reject_safety 0.9 -ts_max_snes_failures 10  -snes_max_it 20 -ts_exact_final_time matchstep
```

#cook membrane

```
../tools/mofem_part -my_file $HOME/Desktop/Meshes/MFront_testing/3D_cook_membrane.cub -my_nparts 10 &&  mpirun -np 10 /home/karol/mofem_install/users_modules/mfront_interface/mfront_interface -file_name out.h5m -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps -ts_max_snes_failures 1 -block_1 LogarithmicStrainPlasticity -param_1_0 450 -param_1_1 130 -snes_rtol 1e-7 -snes_atol 1e-7 -ksp_rtol 1e-12 -ksp_atol 1e-12 -ts_adapt_type none -print_gauss -load_history load_history.in -order 2  -ts_dt 0.01 -ts_adapt_dt_max 0.05  -ts_max_time 1 -snes_monitor -ts_monitor -log_quiet -ts_type theta -ts_adapt_always_accept 1 -ts_theta_initial_guess_extrapolate 1  -ts_theta_theta 1 -ts_adapt_reject_safety 0.9 -ts_max_snes_failures 10  -snes_max_it 20 -ts_exact_final_time matchstep | tee log 
```


tensor convention http://tfel.sourceforge.net/tensors.html

COMMANDS

```
export MGIS_PATH=$(spack find -l --path mgis | awk 'END{print}' | awk 'NF{ print $NF }')"/lib"
```
mfront-query --material-properties Plasticity.mfront



DOWNLOAD BEHAVIOUR

wget http://tfel.sourceforge.net/gallery/plasticity/IsotropicLinearHardeningPlasticity.mfront


BUILD BEHAVIOUR

mfront --obuild --interface=generic IsotropicLinearHardeningPlasticity.mfront
mfront --obuild --interface=generic LogarithmicStrainPlasticity.mfront

mfront --obuild --interface=generic --install-path ./src2/ IsotropicLinearHardeningPlasticity.mfront


LogarithmicStrainPlasticity


export TFELHOME=$SPACKVIEW/lib

export LD_LIBRARY_PATH=$TFELHOME:$LD_LIBRARY_PATH


export LD_LIBRARY_PATH=$TFELHOME/lib:$LD_LIBRARY_PATH

ldd src/libBehaviour.so 



src/libBehaviour.dyld on macOS
