
#include <MoFEM.hpp>

static char help[] = "mfront --obuild --interface=generic BEHAVIOUR.mfront \n";
//  mfront -query-- material-properties BEHAVIOUR.mfront

using namespace MoFEM;
using namespace FTensor;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>

// #ifdef WITH_MFRONT
#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/MaterialDataManager.h"
#include "MGIS/Behaviour/Integrate.hxx"
#include "MGIS/LibrariesManager.hxx"

using namespace mgis;
using namespace mgis::behaviour;

#include <Operators.hpp>

using namespace MFrontInterface;

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface

    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface

    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    int order = 2;
    PetscBool print_gauss = PETSC_FALSE;

    moab::Core mb_post_gauss;
    moab::Interface &moab_gauss = mb_post_gauss;

    auto get_options_from_command_line = [&]() {
      MoFEMFunctionBeginHot;
      CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "none");
      CHKERR PetscOptionsInt("-order", "approximation order", "", order, &order,
                             PETSC_NULL);

      CHKERR PetscOptionsBool("-print_gauss",
                              "print gauss pts (internal variables)", "",
                              print_gauss, &print_gauss, PETSC_NULL);
      ierr = PetscOptionsEnd();
      CHKERRQ(ierr);
      MoFEMFunctionReturnHot(0);
    };

    CHKERR get_options_from_command_line();

    // Add field
    CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    // CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
    // 3);
    CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR simple->setFieldOrder("U", order);
    PetscBool is_partitioned = PETSC_TRUE;

    CHKERR simple->defineFiniteElements();

    // Add Neumann forces elements
    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "U");
    CHKERR MetaNodalForces::addElement(m_field, "U");
    CHKERR MetaEdgeForces::addElement(m_field, "U");

    simple->getOtherFiniteElements().push_back("FORCE_FE");
    simple->getOtherFiniteElements().push_back("PRESSURE_FE");

    CHKERR simple->defineProblem(is_partitioned);
    CHKERR simple->buildFields();
    CHKERR simple->buildFiniteElements();
    CHKERR simple->buildProblem();

    boost::shared_ptr<CommonData> commonDataPtr;
    boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcFe;
    // mofem boundary conditions
    boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    boost::ptr_map<std::string, EdgeForce> edge_forces;
    boost::shared_ptr<DirichletDisplacementBc> dirichlet_bc_ptr;
    boost::shared_ptr<DomainEle> update_int_variables;

    commonDataPtr = boost::make_shared<CommonData>(m_field);
    commonDataPtr->setBlocks();
    commonDataPtr->createTags();

    commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
    commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
    commonDataPtr->mDispPtr = boost::make_shared<MatrixDouble>();
    commonDataPtr->materialTangentPtr = boost::make_shared<MatrixDouble>();
    commonDataPtr->internalVariablePtr = boost::make_shared<MatrixDouble>();

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "none");

    if (commonDataPtr->setOfBlocksData.empty())
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "No blocksets on the mesh has been provided (e.g. MATERIAL1)");

    auto is_lib_finite_strain = [&](const std::string &lib,
                                    const std::string &beh_name) {
      auto &lm = LibrariesManager::get();
      return (lm.getBehaviourType(lib, beh_name) == 2) &&
             (lm.getBehaviourKinematic(lib, beh_name) == 3);
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
      CHKERR PetscOptionsString(
          param_path.c_str(), "path to the behaviour library", "",
          "src/libBehaviour.so", char_name, 255, &is_param);
      if (is_param)
        lib_path = string(char_name);

      auto &mgis_bv_ptr = block.second.mGisBehaviour;
      auto is_finite_strain = is_lib_finite_strain(lib_path, name);
      if (is_finite_strain) {
        mgis_bv_ptr = boost::make_shared<Behaviour>(
            load(op, lib_path, name, Hypothesis::TRIDIMENSIONAL));
        block.second.isFiniteStrain = true;
      } else
        mgis_bv_ptr = boost::make_shared<Behaviour>(
            load(lib_path, name, Hypothesis::TRIDIMENSIONAL));

      CHKERR block.second.setBlockBehaviourData(set_from_blocks);
      for (size_t dd = 0; dd < mgis_bv_ptr->mps.size(); ++dd) {
        double my_param = 0;
        PetscBool is_set = PETSC_FALSE;
        string param_cmd = "-param_" + to_string(id) + "_" + to_string(dd);
        CHKERR PetscOptionsScalar(param_cmd.c_str(), "parameter from cmd", "",
                                  my_param, &my_param, &is_set);
        if (!is_set)
          continue;

        block.second.behDataLHS->s0.material_properties[dd] = my_param;
        block.second.behDataLHS->s1.material_properties[dd] = my_param;
        block.second.behDataRHS->s0.material_properties[dd] = my_param;
        block.second.behDataRHS->s1.material_properties[dd] = my_param;
      }

      int nb = 0;

      // FIXME: PRINT PROPERLY WITH SHOWING WHAT WAS ASSIGNED BY THE USER!!!
      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "%s behaviour loaded on block %d. \n",
                         mgis_bv_ptr->behaviour.c_str(), block.first);
      auto it = block.second.behDataRHS->s1.material_properties.begin();
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

    update_int_variables = boost::make_shared<DomainEle>(m_field);
    auto integration_rule = [&](int, int, int approx_order) {
      return 2 * order + 1;
    };
    update_int_variables->getRuleHook = integration_rule;
    update_int_variables->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>("U", commonDataPtr->mGradPtr));
    if (print_gauss)
      update_int_variables->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>("U", commonDataPtr->mDispPtr));
    for (auto &sit : commonDataPtr->setOfBlocksData) {
      if (sit.second.isFiniteStrain)
        update_int_variables->getOpPtrVector().push_back(
            new OpUpdateVariablesFiniteStrains("U", commonDataPtr, sit.second));
      else
        update_int_variables->getOpPtrVector().push_back(
            new OpUpdateVariablesSmallStrains("U", commonDataPtr, sit.second));
      if (print_gauss)
        update_int_variables->getOpPtrVector().push_back(
            new OpSaveGaussPts("U", moab_gauss, commonDataPtr, sit.second));
    }
    // forces and pressures on surface
    CHKERR MetaNeumannForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       PETSC_NULL, "U");
    // nodal forces
    CHKERR MetaNodalForces::setOperators(m_field, nodal_forces, PETSC_NULL,
                                         "U");
    // edge forces
    CHKERR MetaEdgeForces::setOperators(m_field, edge_forces, PETSC_NULL, "U");

    for (auto mit = neumann_forces.begin(); mit != neumann_forces.end();
         mit++) {
      mit->second->methodsOp.push_back(
          new TimeForceScale("-load_history", true));
    }
    for (auto mit = nodal_forces.begin(); mit != nodal_forces.end(); mit++) {
      mit->second->methodsOp.push_back(
          new TimeForceScale("-load_history", true));
    }
    for (auto mit = edge_forces.begin(); mit != edge_forces.end(); mit++) {
      mit->second->methodsOp.push_back(
          new TimeForceScale("-load_history", true));
    }

    dirichlet_bc_ptr =
        boost::make_shared<DirichletDisplacementBc>(m_field, "U");
    // dirichlet_bc_ptr->dIag = 1;
    dirichlet_bc_ptr->methodsOp.push_back(
        new TimeForceScale("-load_history", true));

    PipelineManager *pipeline_mng = m_field.getInterface<PipelineManager>();
    auto add_domain_base_ops = [&](auto &pipeline) {
      pipeline.push_back(new OpCalculateVectorFieldGradient<3, 3>(
          "U", commonDataPtr->mGradPtr));
    };

    auto add_domain_ops_lhs = [&](auto &pipeline) {
      for (auto &sit : commonDataPtr->setOfBlocksData) {
        if (sit.second.isFiniteStrain) {

          pipeline.push_back(
              new OpTangentFiniteStrains("U", commonDataPtr, sit.second));
          pipeline.push_back(
              new OpAssembleLhsFiniteStrains("U", "U", commonDataPtr));
        } else {

          pipeline.push_back(
              new OpTangentSmallStrains("U", commonDataPtr, sit.second));
          pipeline.push_back(
              new OpAssembleLhsSmallStrains("U", "U", commonDataPtr));
        }
      }
    };

    auto add_domain_ops_rhs = [&](auto &pipeline) {
      for (auto &sit : commonDataPtr->setOfBlocksData)
        if (sit.second.isFiniteStrain)
          pipeline.push_back(
              new OpStressFiniteStrains("U", commonDataPtr, sit.second));

        else
          pipeline.push_back(
              new OpStressSmallStrains("U", commonDataPtr, sit.second));

      pipeline.push_back(new OpAssembleRhs("U", commonDataPtr));
    };

    add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
    add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
    add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
    add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());

    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

    ISManager *is_manager = m_field.getInterface<ISManager>();

    // auto solver = pipeline_mng->createTS();
    auto create_custom_ts = [&]() {
      auto set_dm_section = [&](auto dm) {
        MoFEMFunctionBegin;
        PetscSection section;
        CHKERR m_field.getInterface<ISManager>()->sectionCreate(
            simple->getProblemName(), &section);
        CHKERR DMSetDefaultSection(dm, section);
        CHKERR DMSetDefaultGlobalSection(dm, section);
        CHKERR PetscSectionDestroy(&section);
        MoFEMFunctionReturn(0);
      };

      auto dm = simple->getDM();
      CHKERR set_dm_section(dm);

      boost::shared_ptr<FEMethod> null;
      auto preProc = boost::make_shared<FePrePostProcess>();
      preProc->methodsOp.push_back(new TimeForceScale("-load_history", true));

      // Add element to calculate lhs of stiff part
      if (pipeline_mng->getDomainLhsFE()) {

        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null,
                                     preProc, null);
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null,
                                     dirichlet_bc_ptr, null);

        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(),
                                     pipeline_mng->getDomainLhsFE(), null,
                                     null);
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null, null,
                                     dirichlet_bc_ptr);
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null, null,
                                     preProc);
      }
      if (pipeline_mng->getBoundaryLhsFE())
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getBoundaryFEName(),
                                     pipeline_mng->getBoundaryLhsFE(), null,
                                     null);
      if (pipeline_mng->getSkeletonLhsFE())
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getSkeletonFEName(),
                                     pipeline_mng->getSkeletonLhsFE(), null,
                                     null);

      // Add element to calculate rhs of stiff part
      if (pipeline_mng->getDomainRhsFE()) {

        // add dirichlet boundary conditions
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null,
                                     preProc, null);
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null,
                                     dirichlet_bc_ptr, null);

        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(),
                                     pipeline_mng->getDomainRhsFE(), null,
                                     null);

        // Add surface forces

        for (auto fit = neumann_forces.begin(); fit != neumann_forces.end();
             fit++) {
          CHKERR DMMoFEMTSSetIFunction(dm, fit->first.c_str(),
                                       &fit->second->getLoopFe(), NULL, NULL);
        }

        // Add edge forces
        for (auto fit = edge_forces.begin(); fit != edge_forces.end(); fit++) {
          cerr << fit->first.c_str() << endl;
          CHKERR DMMoFEMTSSetIFunction(dm, fit->first.c_str(),
                                       &fit->second->getLoopFe(), NULL, NULL);
        }

        // Add nodal forces
        for (auto fit = nodal_forces.begin(); fit != nodal_forces.end();
             fit++) {
          CHKERR DMMoFEMTSSetIFunction(dm, fit->first.c_str(),
                                       &fit->second->getLoopFe(), NULL, NULL);
        }

        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null, null,
                                     dirichlet_bc_ptr);
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null, null,
                                     preProc);
      }
      if (pipeline_mng->getBoundaryRhsFE())
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getBoundaryFEName(),
                                     pipeline_mng->getBoundaryRhsFE(), null,
                                     null);
      if (pipeline_mng->getSkeletonRhsFE())
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getSkeletonFEName(),
                                     pipeline_mng->getSkeletonRhsFE(), null,
                                     null);

      auto ts = MoFEM::createTS(m_field.get_comm());
      CHKERR TSSetDM(ts, dm);
      return ts;
    };

    auto solver = create_custom_ts();

    CHKERR TSSetExactFinalTime(solver, TS_EXACTFINALTIME_STEPOVER);
    auto dm = simple->getDM();
    auto D = smartCreateDMVector(dm);

    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR TSSetSolution(solver, D);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);

    auto set_section_monitor = [&]() {
      MoFEMFunctionBegin;
      SNES snes;
      CHKERR TSGetSNES(solver, &snes);
      PetscViewerAndFormat *vf;
      CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                        PETSC_VIEWER_DEFAULT, &vf);
      CHKERR SNESMonitorSet(
          snes,
          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                             void *))SNESMonitorFields,
          vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
      MoFEMFunctionReturn(0);
    };

    auto create_post_process_element = [&]() {
      MoFEMFunctionBegin;
      // postProcFe =
      // boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(m_field);
      postProcFe = boost::make_shared<PostProcVolumeOnRefinedMesh>(m_field);
      postProcFe->generateReferenceElementMesh();

      postProcFe->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("U",
                                                   commonDataPtr->mGradPtr));
      for (auto &sit : commonDataPtr->setOfBlocksData)
        postProcFe->getOpPtrVector().push_back(new OpPostProcElastic(
            "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
            commonDataPtr, sit.second));

      postProcFe->addFieldValuesPostProc("U", "DISPLACEMENT");
      MoFEMFunctionReturn(0);
    };

    auto set_time_monitor = [&]() {
      MoFEMFunctionBegin;
      boost::shared_ptr<Monitor> monitor_ptr(new Monitor(
          dm, postProcFe, update_int_variables, moab_gauss, print_gauss));
      boost::shared_ptr<ForcesAndSourcesCore> null;
      CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                                 monitor_ptr, null, null);
      MoFEMFunctionReturn(0);
    };

    // CHKERR set_section_monitor();
    CHKERR create_post_process_element();
    CHKERR set_time_monitor();

    CHKERR TSSolve(solver, D);

    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
// #endif