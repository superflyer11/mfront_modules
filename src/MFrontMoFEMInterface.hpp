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

#ifndef __MFRONTGENERICINTERFACE_HPP__
#define __MFRONTGENERICINTERFACE_HPP__

using EntData = EntitiesFieldData::EntData;

enum ModelHypothesis { TRIDIMENSIONAL, PLANESTRAIN, AXISYMMETRICAL };

template <ModelHypothesis H> struct MFrontEleType;

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

#ifndef SCHUR_ASSEMBLE
#define SCHUR_ASSEMBLE 0
#endif

constexpr AssemblyType MFRONT_AT =
    (SCHUR_ASSEMBLE) ? AssemblyType::BLOCK_SCHUR
                     : AssemblyType::PETSC; //< selected assembly type

template <ModelHypothesis H = TRIDIMENSIONAL>
struct MFrontMoFEMInterface : public GenericElementInterface {

  using DomainEle = typename MFrontEleType<H>::DomainEle;
  using DomainEleOp = typename MFrontEleType<H>::DomainEleOp;
  using PostProcDomainOnRefinedMesh =
      typename MFrontEleType<H>::PostProcDomainOnRefinedMesh;

  static constexpr int DIM = MFrontEleType<H>::SPACE_DIM;

  using OpInternalForce =
      typename FormsIntegrators<DomainEleOp>::template Assembly<MFRONT_AT>::
          template LinearForm<GAUSS>::template OpGradTimesTensor<1, DIM, DIM>;
  using OpAssembleLhsFiniteStrains =
      typename FormsIntegrators<DomainEleOp>::template Assembly<MFRONT_AT>::
          template BiLinearForm<GAUSS>::template OpGradTensorGrad<1, DIM, DIM,
                                                                  1>;
  using OpAssembleLhsSmallStrains =
      typename FormsIntegrators<DomainEleOp>::template Assembly<MFRONT_AT>::
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

  boost::shared_ptr<PostProcDomainOnRefinedMesh> postProcFe;
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
                       PetscBool is_quasi_static = PETSC_TRUE,
                       PetscInt order = -1);

  MoFEMErrorCode getCommandLineParameters() override;
  MoFEMErrorCode addElementFields() override;
  MoFEMErrorCode createElements() override;
  MoFEMErrorCode setOperators() override;

  MoFEMErrorCode
  setIntegrationRule(boost::shared_ptr<ForcesAndSourcesCore> fe_ptr);
  MoFEMErrorCode setBaseOperators(
      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip);

  MoFEMErrorCode setUpdateElementVariablesOperators() override;

  MoFEMErrorCode opFactoryDomainRhs(
      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip) override;
  MoFEMErrorCode opFactoryDomainLhs(
      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip) override;

  MoFEMErrorCode testOperators();

  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) override;

  MoFEMErrorCode setupSolverJacobianTS(const TSType type) override;
  MoFEMErrorCode setupSolverFunctionTS(const TSType type) override;

  MoFEMErrorCode setupSolverJacobianSNES() override;
  MoFEMErrorCode setupSolverFunctionSNES() override;

  MoFEMErrorCode updateElementVariables(SmartPetscObj<DM> dm,
                                        string fe_name) override;
  MoFEMErrorCode updateElementVariables() override {
    return updateElementVariables(dM, "MFRONT_EL");
  }

  MoFEMErrorCode postProcessElement(int step, SmartPetscObj<DM> dm,
                                    string fe_name) override;
  MoFEMErrorCode postProcessElement(int step) override {
    return postProcessElement(step, dM, "MFRONT_EL");
  }
};

#endif // __MFRONTGENERICINTERFACE_HPP__