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

extern double mfront_dt;
extern double mfront_dt_prop;

using EntData = EntitiesFieldData::EntData;

// using EntData = EntitiesFieldData::EntData;
// using DomainEle = FaceElementForcesAndSourcesCore;
// using DomainEleOp = DomainEle::UserDataOperator;

// template <int DIM>
// using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<
//     PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, DIM, DIM>;
// template <int DIM>
// using OpAssembleLhsFiniteStrains = FormsIntegrators<DomainEleOp>::Assembly<
//     PETSC>::BiLinearForm<GAUSS>::OpGradTensorGrad<1, DIM, DIM, 1>;
// template <int DIM>
// using OpAssembleLhsSmallStrains = FormsIntegrators<DomainEleOp>::Assembly<
//     PETSC>::BiLinearForm<GAUSS>::OpGradSymTensorGrad<1, DIM, DIM, 0>;

enum hypothesis { TRIDIMENSIONAL, PLANESTRAIN, AXISYMMETRICAL};

template <int H> struct MFrontEleType;

template <> struct MFrontEleType<TRIDIMENSIONAL> {

  MFrontEleType() = delete;
  ~MFrontEleType() = delete;

  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcDomainOnRefinedMesh = PostProcVolumeOnRefinedMesh;

  static constexpr int SPACE_DIM = 3;
};

template <> struct MFrontEleType<PLANESTRAIN> {

  MFrontEleType() = delete;
  ~MFrontEleType() = delete;

  using DomainEle = FaceElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcDomainOnRefinedMesh = PostProcFaceOnRefinedMesh;
  
  static constexpr int SPACE_DIM = 2;
};

template <> struct MFrontEleType<AXISYMMETRICAL> {

  MFrontEleType() = delete;
  ~MFrontEleType() = delete;

  using DomainEle = FaceElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcDomainOnRefinedMesh = PostProcFaceOnRefinedMesh;

  static constexpr int SPACE_DIM = 2;
};

// constexpr int MFrontEleType<PLANE_STEESS>::SPACE_DIM;

template <int H>
struct MFrontMoFEMInterface : public GenericElementInterface {

  using DomainEle = typename MFrontEleType<H>::DomainEle;
  using DomainEleOp = typename DomainEle::UserDataOperator;
  static constexpr int DIM = MFrontEleType<H>::SPACE_DIM;

  using OpInternalForce =
      typename FormsIntegrators<DomainEleOp>::template Assembly<PETSC>::
          template LinearForm<GAUSS>::template OpGradTimesTensor<1, DIM, DIM>;
  using OpAssembleLhsFiniteStrains =
      typename FormsIntegrators<DomainEleOp>::template Assembly<PETSC>::
          template BiLinearForm<GAUSS>::template OpGradTensorGrad<1, DIM, DIM,
                                                                  1>;
  using OpAssembleLhsSmallStrains =
      typename FormsIntegrators<DomainEleOp>::template Assembly<PETSC>::
          template BiLinearForm<GAUSS>::template OpGradSymTensorGrad<1, DIM,
                                                                     DIM, 0>;

  MoFEM::Interface &mField;
  string optionsPrefix;

  SmartPetscObj<DM> dM;

  PetscBool isQuasiStatic;
  PetscBool saveGauss;
  PetscBool saveVolume;

  PetscBool testJacobian;
  PetscReal randomFieldScale;

  PetscInt oRder;
  bool isDisplacementField;
  bool isFiniteKinematics;
  BitRefLevel bIt;

  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcFe;
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

  // FIXME: Add this funtion to GenericElementInterface
  MoFEMErrorCode testOperators();

  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) override;

  MoFEMErrorCode setupSolverJacobianTS(const TSType type) override;
  MoFEMErrorCode setupSolverFunctionTS(const TSType type) override;

  MoFEMErrorCode setupSolverJacobianSNES() override;
  MoFEMErrorCode setupSolverFunctionSNES() override;

  MoFEMErrorCode updateElementVariables() override;
  MoFEMErrorCode postProcessElement(int step) override;

  // template <bool UPDATE, bool IS_LARGE_STRAIN, int D> struct OpStressTmp;
  // template <typename T, int D> struct OpTangent;
  // struct OpPostProcElastic;
  // struct OpPostProcInternalVariables;
  // struct OpSaveGaussPts;
};

#endif // __MFRONTGENERICINTERFACE_HPP__