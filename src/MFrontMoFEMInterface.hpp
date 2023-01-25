/** \file MFrontMoFEMInterface.hpp
 * @brief
 * @date 2023-01-25
 *
 * @copyright Copyright (c) 202
 *
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
  int atomTest;
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

  MoFEMErrorCode getCommandLineParameters() override;
  MoFEMErrorCode addElementFields() override;
  MoFEMErrorCode createElements() override;
  MoFEMErrorCode setOperators() override;
  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) override;

  MoFEMErrorCode setupSolverJacobianTS(const TSType type) override;
  MoFEMErrorCode setupSolverFunctionTS(const TSType type) override;

  MoFEMErrorCode setupSolverJacobianSNES() override;
  MoFEMErrorCode setupSolverFunctionSNES() override;

  MoFEMErrorCode updateElementVariables() override;
  MoFEMErrorCode postProcessElement(int step) override;
};

#endif // __MFRONTGENERICINTERFACE_HPP__