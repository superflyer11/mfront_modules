/** \file MFrontMoFEMInterface.hpp
 */

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the MIT License>. */

#pragma once

#ifndef __MFRONTGENERICINTERFACE_HPP__
#define __MFRONTGENERICINTERFACE_HPP__

struct MFrontMoFEMInterface : public GenericElementInterface {

  using EntData = EntitiesFieldData::EntData;
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;

  using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<
      PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 3, 3>;
  using OpAssembleLhsFiniteStrains = FormsIntegrators<DomainEleOp>::Assembly<
      PETSC>::BiLinearForm<GAUSS>::OpGradTensorGrad<1, 3, 3, 1>;
  using OpAssembleLhsSmallStrains = FormsIntegrators<DomainEleOp>::Assembly<
      PETSC>::BiLinearForm<GAUSS>::OpGradSymTensorGrad<1, 3, 3, 0>;
  
  MoFEM::Interface &mField;
  string optionsPrefix;

  SmartPetscObj<DM> dM;

  PetscBool isQuasiStatic;
  PetscBool printGauss;

  PetscInt oRder;
  bool isDisplacementField;
  bool isFiniteKinematics;
  BitRefLevel bIt;
  
  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcFe;
  boost::shared_ptr<DomainEle> updateIntVariablesElePtr;

  boost::shared_ptr<DomainEle> mfrontPipelineRhsPtr;
  boost::shared_ptr<DomainEle> mfrontPipelineLhsPtr;

  string positionField;
  string meshNodeField;

//   moab::Core mb_postGauss;
  boost::shared_ptr<moab::Interface> moabGaussIntPtr;

  MFrontMoFEMInterface(MoFEM::Interface &m_field, string postion_field = "U",
                       string mesh_posi_field_name = "MESH_NODE_POSITIONS",
                       bool is_displacement_field = true,
                       PetscBool is_quasi_static = PETSC_TRUE);

  MoFEMErrorCode getCommandLineParameters();
  MoFEMErrorCode addElementFields();
  MoFEMErrorCode createElements();
  MoFEMErrorCode setOperators();
  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm);

  MoFEMErrorCode setupSolverJacobianTS(const TSType type);
  MoFEMErrorCode setupSolverFunctionTS(const TSType type);

  MoFEMErrorCode setupSolverJacobianSNES();
  MoFEMErrorCode setupSolverFunctionSNES();

  MoFEMErrorCode updateElementVariables();
  MoFEMErrorCode postProcessElement(int step);
};

#endif // __MFRONTGENERICINTERFACE_HPP__