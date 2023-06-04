/** \file MFrontMoFEMInterface.cpp
 * @brief
 * @date 2023-01-25
 *
 * @copyright Copyright (c) 202
 *
 */

#include <MoFEM.hpp>

using namespace MoFEM;
using namespace FTensor;

#include <BasicFiniteElements.hpp>
#include <quad.h>
#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/Integrate.hxx"
#include "MGIS/LibrariesManager.hxx"

using namespace mgis;
using namespace mgis::behaviour;

#include <MFrontMoFEMInterface.hpp>

#include <MFrontOperators.hpp>
using namespace MFrontInterface;

template struct MFrontMoFEMInterface<TRIDIMENSIONAL>;
template struct MFrontMoFEMInterface<AXISYMMETRICAL>;
template struct MFrontMoFEMInterface<PLANESTRAIN>;

template <ModelHypothesis H>
MFrontMoFEMInterface<H>::MFrontMoFEMInterface(MoFEM::Interface &m_field,
                                              string postion_field,
                                              string mesh_posi_field_name,
                                              bool is_displacement_field,
                                              PetscBool is_quasi_static)
    : mField(m_field), positionField(postion_field),
      meshNodeField(mesh_posi_field_name),
      isDisplacementField(is_displacement_field),
      isQuasiStatic(is_quasi_static) {
  oRder = -1;
  isFiniteKinematics = true;
  saveGauss = PETSC_FALSE;
  saveVolume = PETSC_TRUE;
  testJacobian = PETSC_FALSE;
  randomFieldScale = 1.0;
  optionsPrefix = "mi_";
}

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::getCommandLineParameters() {
  MoFEMFunctionBegin;
  isQuasiStatic = PETSC_FALSE;
  oRder = 2;
  CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, optionsPrefix.c_str(), "", "none");
  // FIXME: make it possible to set a separate orders for contact, nonlinear
  // elasticity, mfront... etc
  CHKERR PetscOptionsInt("-order", "approximation order", "", oRder, &oRder,
                         PETSC_NULL);

  CHKERR PetscOptionsBool("-save_gauss", "save gauss pts (internal variables)",
                          "", saveGauss, &saveGauss, PETSC_NULL);
  CHKERR PetscOptionsBool("-save_volume", "save results on a volumetric mesh",
                          "", saveVolume, &saveVolume, PETSC_NULL);

  CHKERR PetscOptionsBool("-test_jacobian", "test Jacobian (LHS matrix)", "",
                          testJacobian, &testJacobian, PETSC_NULL);
  CHKERR PetscOptionsReal("-random_field_scale",
                          "scale for the finite difference jacobian", "",
                          randomFieldScale, &randomFieldScale, PETSC_NULL);

  if (saveGauss)
    moabGaussIntPtr = boost::shared_ptr<moab::Interface>(new moab::Core());

  commonDataPtr = boost::make_shared<CommonData>(mField);
  commonDataPtr->setBlocks(DIM);
  commonDataPtr->createTags();

  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mFullStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDispPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mPrevGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mPrevStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->materialTangentPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mFullTangentPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->internalVariablePtr = boost::make_shared<MatrixDouble>();

  if (commonDataPtr->setOfBlocksData.empty())
    MOFEM_LOG("WORLD", Sev::inform)
        << "No blocksets on the mesh has been provided for MFront (e.g. "
           "MFRONT_MAT_1)";

  auto check_lib_finite_strain = [&](const std::string &lib,
                                     const std::string &beh_name, bool &flag) {
    MoFEMFunctionBeginHot;

    ifstream f(lib.c_str());
    if (!f.good())
      MOFEM_LOG("WORLD", Sev::error)
          << "Problem with the behaviour path: " << lib;

    auto &lm = LibrariesManager::get();
    flag = bool(lm.getBehaviourType(lib, beh_name) == 2) &&
           (lm.getBehaviourKinematic(lib, beh_name) == 3);
    MoFEMFunctionReturnHot(0);
  };

  auto op = FiniteStrainBehaviourOptions{};
  op.stress_measure = FiniteStrainBehaviourOptions::PK1;
  op.tangent_operator = FiniteStrainBehaviourOptions::DPK1_DF;

  for (auto &block : commonDataPtr->setOfBlocksData) {
    const int &id = block.first;
    auto &lib_path = block.second.behaviourPath;
    auto &name = block.second.behaviourName;
    const string param_name = "-block_" + to_string(id);
    const string param_path = "-lib_path_" + to_string(id);
    const string param_from_blocks = "-my_params_" + to_string(id);
    PetscBool set_from_blocks = PETSC_FALSE;
    char char_name[255];
    PetscBool is_param;

    CHKERR PetscOptionsBool(param_from_blocks.c_str(),
                            "set parameters from blocks", "", set_from_blocks,
                            &set_from_blocks, PETSC_NULL);

    CHKERR PetscOptionsString(param_name.c_str(), "name of the behaviour", "",
                              "IsotropicLinearHardeningPlasticity", char_name,
                              255, &is_param);
    if (is_param)
      name = string(char_name);
    string default_lib_path =
        "src/libBehaviour." + string(DEFAULT_LIB_EXTENSION);
    CHKERR PetscOptionsString(
        param_path.c_str(), "path to the behaviour library", "",
        default_lib_path.c_str(), char_name, 255, &is_param);
    if (is_param)
      lib_path = string(char_name);
    auto &mgis_bv_ptr = block.second.mGisBehaviour;
    bool is_finite_strain = false;

    CHKERR check_lib_finite_strain(lib_path, name, is_finite_strain);

    mgis::behaviour::Hypothesis h;
    switch (H) {
    case TRIDIMENSIONAL:
      h = mgis::behaviour::Hypothesis::TRIDIMENSIONAL;
      break;
    case PLANESTRAIN:
      h = mgis::behaviour::Hypothesis::PLANESTRAIN;
      break;
    case AXISYMMETRICAL:
      h = mgis::behaviour::Hypothesis::AXISYMMETRICAL;
      break;
    default:
      break;
    }

    if (is_finite_strain) {
      mgis_bv_ptr = boost::make_shared<Behaviour>(load(op, lib_path, name, h));
      block.second.isFiniteStrain = true;
    } else
      mgis_bv_ptr = boost::make_shared<Behaviour>(load(lib_path, name, h));

    CHKERR block.second.setBlockBehaviourData(set_from_blocks);
    for (size_t dd = 0; dd < mgis_bv_ptr->mps.size(); ++dd) {
      double my_param = 0;
      PetscBool is_set = PETSC_FALSE;
      string param_cmd = "-param_" + to_string(id) + "_" + to_string(dd);
      CHKERR PetscOptionsScalar(param_cmd.c_str(), "parameter from cmd", "",
                                my_param, &my_param, &is_set);
      if (!is_set)
        continue;
      setMaterialProperty(block.second.behDataPtr->s0, dd, my_param);
      setMaterialProperty(block.second.behDataPtr->s1, dd, my_param);
    }

    int nb = 0;

    // PRINT PROPERLY WITH SHOWING WHAT WAS ASSIGNED BY THE USER!!!
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "%s behaviour loaded on block %d. \n",
                       mgis_bv_ptr->behaviour.c_str(), block.first);
    if (is_finite_strain)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Finite Strain Kinematics \n");
    else
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Small Strain Kinematics \n");

    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Internal variables: \n");
    for (const auto &is : mgis_bv_ptr->isvs)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, ": %s\n", is.name.c_str());
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "External variables: \n");
    for (const auto &es : mgis_bv_ptr->esvs)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, ": %s\n", es.name.c_str());

    auto it = block.second.behDataPtr->s0.material_properties.begin();
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Material properties: \n");
    for (const auto &mp : mgis_bv_ptr->mps)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "%d : %s = %g\n", nb++,
                         mp.name.c_str(), *it++);

    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Real parameters: \n");
    for (auto &p : mgis_bv_ptr->params)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "%d : %s\n", nb++, p.c_str());
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Integer parameters: \n");
    for (auto &p : mgis_bv_ptr->iparams)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "%d : %s\n", nb++, p.c_str());
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Unsigned short parameters: \n");
    for (auto &p : mgis_bv_ptr->usparams)
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "%d : %s\n", nb++, p.c_str());
  }

  ierr = PetscOptionsEnd();
  CHKERRQ(ierr);

  auto check_behaviours_kinematics = [&](bool &is_finite_kin) {
    MoFEMFunctionBeginHot;
    is_finite_kin =
        commonDataPtr->setOfBlocksData.begin()->second.isFiniteStrain;
    for (auto &block : commonDataPtr->setOfBlocksData) {
      if (block.second.isFiniteStrain != is_finite_kin)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "All used MFront behaviours have to be of same kinematics "
                "(small or "
                "large strains)");
    }
    MoFEMFunctionReturnHot(0);
  };

  CHKERR check_behaviours_kinematics(isFiniteKinematics);

  MoFEMFunctionReturn(0);
};

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::addElementFields() {
  MoFEMFunctionBeginHot;
  auto simple = mField.getInterface<Simple>();
  if (!mField.check_field(positionField)) {
    CHKERR simple->addDomainField(positionField, H1, AINSWORTH_LEGENDRE_BASE,
                                  DIM);
    CHKERR simple->addBoundaryField(positionField, H1, AINSWORTH_LEGENDRE_BASE,
                                    DIM);
    CHKERR simple->setFieldOrder(positionField, oRder);
  }
  if (!mField.check_field(meshNodeField)) {
    CHKERR simple->addDataField(meshNodeField, H1, AINSWORTH_LEGENDRE_BASE,
                                DIM);
    CHKERR simple->setFieldOrder(meshNodeField, 2);
  }

  MoFEMFunctionReturnHot(0);
};

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::createElements() {
  MoFEMFunctionBeginHot;

  CHKERR mField.add_finite_element("MFRONT_EL", MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row("MFRONT_EL", positionField);
  CHKERR mField.modify_finite_element_add_field_col("MFRONT_EL", positionField);
  CHKERR mField.modify_finite_element_add_field_data("MFRONT_EL",
                                                     positionField);
  if (mField.check_field(meshNodeField))
    CHKERR mField.modify_finite_element_add_field_data("MFRONT_EL",
                                                       meshNodeField);

  mfrontPipelineRhsPtr = boost::make_shared<DomainEle>(mField);
  mfrontPipelineLhsPtr = boost::make_shared<DomainEle>(mField);
  updateIntVariablesElePtr = boost::make_shared<DomainEle>(mField);

  // FIXME
  //  TODO: update it according to the newest MoFEM
  //  CHKERR addHOOpsVol(meshNodeField, *mfrontPipelineRhsPtr, true, false,
  //  false,
  //                     false);
  //  CHKERR addHOOpsVol(meshNodeField, *mfrontPipelineLhsPtr, true, false,
  //  false,
  //                     false);
  //  CHKERR addHOOpsVol(meshNodeField, *updateIntVariablesElePtr, true, false,
  //                     false, false);

  CHKERR AddHOOps<DIM, DIM, DIM>::add(mfrontPipelineRhsPtr->getOpPtrVector(),
                                      {H1}, meshNodeField);
  CHKERR AddHOOps<DIM, DIM, DIM>::add(mfrontPipelineLhsPtr->getOpPtrVector(),
                                      {H1}, meshNodeField);
  CHKERR AddHOOps<DIM, DIM, DIM>::add(
      updateIntVariablesElePtr->getOpPtrVector(), {H1}, meshNodeField);

  for (auto &[id, data] : commonDataPtr->setOfBlocksData) {
    CHKERR mField.add_ents_to_finite_element_by_dim(data.tEts, DIM,
                                                    "MFRONT_EL");
    // if (oRder > 0) {

    //   CHKERR mField.set_field_order(data.tEts.subset_by_dimension(1),
    //                                 positionField, oRder);
    //   CHKERR
    //   mField.set_field_order(data.tEts.subset_by_dimension(2), positionField,
    //                          oRder);
    //   CHKERR
    //   mField.set_field_order(data.tEts.subset_by_dimension(3), positionField,
    //                          oRder);
    // }
  }

  MoFEMFunctionReturnHot(0);
};

template <ModelHypothesis H> MoFEMErrorCode MFrontMoFEMInterface<H>::setOperators() {
  MoFEMFunctionBegin;

  auto &moab_gauss = *moabGaussIntPtr;

  auto integration_rule = [&](int, int, int approx_order) {
    return 2 * approx_order + 1;
  };

  mfrontPipelineLhsPtr->getRuleHook = integration_rule;
  mfrontPipelineRhsPtr->getRuleHook = integration_rule;
  updateIntVariablesElePtr->getRuleHook = integration_rule;

  updateIntVariablesElePtr->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<DIM, DIM>(positionField,
                                                   commonDataPtr->mGradPtr));
  updateIntVariablesElePtr->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<DIM>(positionField,
                                            commonDataPtr->mDispPtr));
  if (isFiniteKinematics)
    updateIntVariablesElePtr->getOpPtrVector().push_back(
        new OpUpdateVariablesFiniteStrains<H>(positionField, commonDataPtr));
  else
    updateIntVariablesElePtr->getOpPtrVector().push_back(
        new OpUpdateVariablesSmallStrains<H>(positionField, commonDataPtr));
  // if (saveGauss)
  //   updateIntVariablesElePtr->getOpPtrVector().push_back(
  //       new OpSaveGaussPts(positionField, moab_gauss, commonDataPtr));

  auto jacobian = [&](const double r, const double, const double) {
    if (H == AXISYMMETRICAL)
      return 2. * M_PI * r;
    else
      return 1.;
  };

  auto add_domain_ops_lhs = [&](auto &pipeline) {
    if (isFiniteKinematics) {
      pipeline.push_back(
          new OpTangentFiniteStrains<DIM, H>(positionField, commonDataPtr));
      pipeline.push_back(new OpAssembleLhsFiniteStrains(
          positionField, positionField, commonDataPtr->materialTangentPtr,
          jacobian));
    } else {
      pipeline.push_back(
          new OpTangentSmallStrains<DIM, H>(positionField, commonDataPtr));
      pipeline.push_back(new OpAssembleLhsSmallStrains(
          positionField, positionField, commonDataPtr->materialTangentPtr,
          nullptr, jacobian));
    }
    if (H == AXISYMMETRICAL)
      pipeline.push_back(new OpAxisymmetricLhs(positionField, commonDataPtr));
  };

  auto add_domain_ops_rhs = [&](auto &pipeline) {
    if (isFiniteKinematics)
      pipeline.push_back(
          new OpStressFiniteStrains<H>(positionField, commonDataPtr));
    else
      pipeline.push_back(
          new OpStressSmallStrains<H>(positionField, commonDataPtr));

    pipeline.push_back(new OpInternalForce(
        positionField, commonDataPtr->mStressPtr, jacobian));

    if (H == AXISYMMETRICAL)
      pipeline.push_back(new OpAxisymmetricRhs(positionField, commonDataPtr));
  };

  auto add_domain_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpCalculateVectorFieldValues<DIM>(
        positionField, commonDataPtr->mDispPtr));
    pipeline.push_back(new OpCalculateVectorFieldGradient<DIM, DIM>(
        positionField, commonDataPtr->mGradPtr));
  };

  add_domain_base_ops(mfrontPipelineLhsPtr->getOpPtrVector());
  add_domain_base_ops(mfrontPipelineRhsPtr->getOpPtrVector());

  add_domain_ops_lhs(mfrontPipelineLhsPtr->getOpPtrVector());
  add_domain_ops_rhs(mfrontPipelineRhsPtr->getOpPtrVector());

  // if (DIM == 3 && testJacobian) {
  //   // FIXME: implement 2D
  //   CHKERR testOperators();
  // }

  MoFEMFunctionReturn(0);
}

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::testOperators() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto opt = mField.getInterface<OperatorsTester>();

  Range tets, verts, ho_ents;
  for (auto &[id, data] : commonDataPtr->setOfBlocksData) {
    tets.merge(data.tEts);
  }

  CHKERR mField.get_moab().get_connectivity(tets, verts, true);
  for (auto d : {1, 2, 3}) {
    CHKERR mField.get_moab().get_adjacencies(verts, d, false, ho_ents,
                                             moab::Interface::UNION);
  }

  auto set_random_field = [&](double scale_verts, double scale_ho_ents) {
    auto x =
        opt->setRandomFields(dM, {{positionField, {-scale_verts, scale_verts}}},
                             boost::make_shared<Range>(verts));

    auto x_ho_ents = opt->setRandomFields(
        dM, {{positionField, {-scale_ho_ents, scale_ho_ents}}},
        boost::make_shared<Range>(ho_ents));

    CHKERR VecAXPY(x, 1., x_ho_ents);
    CHKERR VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);

    return x;
  };

  auto x0 = set_random_field(randomFieldScale, randomFieldScale * 0.1);

  CHKERR DMoFEMMeshToLocalVector(dM, x0, INSERT_VALUES, SCATTER_REVERSE);
  CHKERR updateElementVariables();

  auto x = set_random_field(randomFieldScale, randomFieldScale * 0.1);
  auto diff_x = set_random_field(randomFieldScale, randomFieldScale * 0.1);

  // auto post_proc = [&](auto dm, auto f_res, auto out_name) {
  //   MoFEMFunctionBegin;
  //   auto post_proc_fe =
  //       boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(mField);

  //   using OpPPMap = OpPostProcMapInMoab<3, 3>;

  //   auto l_mat = boost::make_shared<MatrixDouble>();
  //   post_proc_fe->getOpPtrVector().push_back(
  //       new OpCalculateVectorFieldValues<3>(positionField, l_mat, f_res));

  //   post_proc_fe->getOpPtrVector().push_back(

  //       new OpPPMap(

  //           post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

  //           {},

  //           {{positionField, l_mat}},

  //           {}, {})

  //   );

  //   CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
  //                                   post_proc_fe);

  //   post_proc_fe->writeFile(out_name);
  //   MoFEMFunctionReturn(0);
  // };

  // CHKERR post_proc(dM, x0, "x0.h5m");
  // CHKERR post_proc(dM, x, "x.h5m");
  // CHKERR post_proc(dM, diff_x, "x_diff.h5m");

  auto res = opt->assembleVec(
      dM, simple->getDomainFEName(), mfrontPipelineRhsPtr, x,
      SmartPetscObj<Vec>(), SmartPetscObj<Vec>(), 0, 1, CacheTupleSharedPtr());

  double res_norm;
  CHKERR VecNorm(res, NORM_2, &res_norm);

  double eps = res_norm * 1e-10;

  auto diff_res = opt->checkCentralFiniteDifference(
      dM, simple->getDomainFEName(), mfrontPipelineRhsPtr, mfrontPipelineLhsPtr,
      x, SmartPetscObj<Vec>(), SmartPetscObj<Vec>(), diff_x, 0, 1, eps);

  double diff_res_norm;
  CHKERR VecNorm(diff_res, NORM_2, &diff_res_norm);

  double rel_diff = diff_res_norm / res_norm;
  MOFEM_LOG_C("WORLD", Sev::inform,
              "Relative difference between hand-coded and finite difference "
              "Jacobian: %3.4e",
              rel_diff);

  constexpr double err = 1e-7;
  if (rel_diff > err)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
            "Relative norm of the difference between hand-coded and the "
            "finite difference Jacobian is too high");

  auto zero = set_random_field(0, 0);
  CHKERR DMoFEMMeshToLocalVector(dM, zero, INSERT_VALUES, SCATTER_REVERSE);
  commonDataPtr->clearTags();

  MoFEMFunctionReturn(0);
}

// BitRefLevel MFrontMoFEMInterface::getBitRefLevel() { return bIt; };

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::addElementsToDM(SmartPetscObj<DM> dm) {
  MoFEMFunctionBeginHot;
  this->dM = dm;
  CHKERR DMMoFEMAddElement(dM, "MFRONT_EL");
  mField.getInterface<Simple>()->getOtherFiniteElements().push_back(
      "MFRONT_EL");

  MoFEMFunctionReturnHot(0);
};

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::setupSolverJacobianSNES() {
  MoFEMFunctionBegin;
  CHKERR DMMoFEMSNESSetJacobian(dM, "MFRONT_EL", mfrontPipelineLhsPtr.get(),
                                NULL, NULL);

  MoFEMFunctionReturn(0);
};

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::setupSolverFunctionSNES() {
  MoFEMFunctionBegin;
  CHKERR DMMoFEMSNESSetFunction(dM, "MFRONT_EL", mfrontPipelineRhsPtr.get(),
                                PETSC_NULL, PETSC_NULL);
  MoFEMFunctionReturn(0);
};

template <ModelHypothesis H>
MoFEMErrorCode
MFrontMoFEMInterface<H>::setupSolverJacobianTS(const TSType type) {
  MoFEMFunctionBegin;
  auto &method = mfrontPipelineLhsPtr;
  switch (type) {
  case IM:
    CHKERR DMMoFEMTSSetIJacobian(dM, "MFRONT_EL", method, method, method);
    break;
  case IM2:
    CHKERR DMMoFEMTSSetI2Jacobian(dM, "MFRONT_EL", method, method, method);
    break;
  case EX:
    CHKERR DMMoFEMTSSetRHSJacobian(dM, "MFRONT_EL", method, method, method);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "This TS is not yet implemented");
    break;
  }
  MoFEMFunctionReturn(0);
};

template <ModelHypothesis H>
MoFEMErrorCode
MFrontMoFEMInterface<H>::setupSolverFunctionTS(const TSType type) {
  MoFEMFunctionBegin;
  auto &method = mfrontPipelineRhsPtr;
  switch (type) {
  case IM:
    CHKERR DMMoFEMTSSetIFunction(dM, "MFRONT_EL", method, method, method);
    break;
  case IM2:
    CHKERR DMMoFEMTSSetI2Function(dM, "MFRONT_EL", method, method, method);
    break;
  case EX:
    CHKERR DMMoFEMTSSetRHSFunction(dM, "MFRONT_EL", method, method, method);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
    break;
  }

  MoFEMFunctionReturn(0);
};

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::postProcessElement(int step) {
  MoFEMFunctionBegin;

  auto create_post_process_element = [&]() {
    MoFEMFunctionBegin;
    
    postProcFe = boost::make_shared<PostProcDomainOnRefinedMesh>(mField);

    postProcFe->generateReferenceElementMesh();

    // postProcFe->getOpPtrVector().push_back(
    //     new OpCalculateVectorFieldGradient<DIM, DIM>(positionField,
    //                                                  commonDataPtr->mGradPtr));
    // postProcFe->getOpPtrVector().push_back(
    //     new OpPostProcElastic(positionField, postProcFe->postProcMesh,
    //                           postProcFe->mapGaussPts, commonDataPtr));

    // FIXME: projection is not working correctly
    // FIXME: pushing this operator leads to MFront integration failure
    // int rule = 2 * oRder + 1;
    // postProcFe->getOpPtrVector().push_back(new OpPostProcInternalVariables(
    //     positionField, postProcFe->postProcMesh, postProcFe->mapGaussPts,
    //     commonDataPtr, rule));

    postProcFe->addFieldValuesPostProc(positionField, "DISPLACEMENT");
    MoFEMFunctionReturn(0);
  };

  if (!postProcFe)
    CHKERR create_post_process_element();

  auto make_vtks = [&]() {
    MoFEMFunctionBegin;

    if (saveVolume) {
      CHKERR DMoFEMLoopFiniteElements(dM, "MFRONT_EL", postProcFe);
      CHKERR postProcFe->writeFile("out_" + optionsPrefix +
                                   boost::lexical_cast<std::string>(step) +
                                   ".h5m");
    }

    if (saveGauss) {
      string file_name = "out_" + optionsPrefix + "gauss_" +
                         boost::lexical_cast<std::string>(step) + ".h5m";

      CHKERR moabGaussIntPtr->write_file(file_name.c_str(), "MOAB",
                                         "PARALLEL=WRITE_PART");
      CHKERR moabGaussIntPtr->delete_mesh();
    }

    MoFEMFunctionReturn(0);
  };

  CHKERR make_vtks();

  MoFEMFunctionReturn(0);
};

template <ModelHypothesis H>
MoFEMErrorCode MFrontMoFEMInterface<H>::updateElementVariables() {

  MoFEMFunctionBegin;
  CHKERR DMoFEMLoopFiniteElements(dM, "MFRONT_EL", updateIntVariablesElePtr);
  MoFEMFunctionReturn(0);
};