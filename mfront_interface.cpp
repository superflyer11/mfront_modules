
#include <MoFEM.hpp>

static char help[] = "mfront --obuild --interface=generic BEHAVIOUR.mfront \n";
//  mfront -query-- material-properties BEHAVIOUR.mfront

using namespace MoFEM;
using namespace FTensor;

using EntData = EntitiesFieldData::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>
#include <quad.h>

#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/MaterialDataManager.h"
#include "MGIS/Behaviour/Integrate.hxx"
#include "MGIS/LibrariesManager.hxx"

using namespace mgis;
using namespace mgis::behaviour;

#include <Operators.hpp>

using namespace MFrontInterface;

#include <ElasticMaterials.hpp>
#include <NonlinearElasticElementInterface.hpp>

#include <BasicBoundaryConditionsInterface.hpp>
#include <SurfacePressureComplexForLazy.hpp>

// #ifdef WITH_MODULE_MFRONT_INTERFACE
#include <MFrontMoFEMInterface.hpp>
// #endif

using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = FaceElementForcesAndSourcesCore;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcVolumeOnRefinedMesh;
using PostProcSkinEle = PostProcFaceOnRefinedMesh;

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-ksp_atol 1e-10 \n"
                                 "-ksp_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_linesearch_type bt \n"
                                 "-snes_linesearch_max_it 3 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-ts_monitor \n"
                                 "-ts_alpha_radius 1 \n"
                                 "-ts_monitor \n"
                                 "-mat_mumps_icntl_20 0 \n"
                                 "-mat_mumps_icntl_14 1200 \n"
                                 "-mat_mumps_icntl_24 1 \n"
                                 "-mat_mumps_icntl_13 1 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(),
                                            "MoFEM_MFront_Interface"));
  LogManager::setLog("MoFEM_MFront_Interface");
  MOFEM_LOG_TAG("MoFEM_MFront_Interface", "module_manager");

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    int order;
    int save_every_nth_step = 1;
    PetscBool is_quasi_static = PETSC_TRUE;
    PetscBool is_partitioned = PETSC_TRUE;

    SmartPetscObj<TS> tSolver;
    SmartPetscObj<DM> dM;

    boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
    boost::shared_ptr<FEMethod> monitor_ptr;

    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_quasi_static", &is_quasi_static,
                               PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "-order", &order, PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "-output_every", &save_every_nth_step,
                              PETSC_NULL);

    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_partitioned", &is_partitioned,
                               PETSC_NULL);
    MOFEM_LOG("WORLD", Sev::inform)
        << "Mesh Partition Flag Status: " << is_partitioned;

    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();

    if (is_partitioned)
      CHKERR simple->loadFile("");
    else
      CHKERR simple->loadFile("", "");

    simple->getProblemName() = "MoFEM MFront Interface module";
    simple->getDomainFEName() = "MFRONT_EL";

    FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;
    // Add displacement field 
    CHKERR m_field.add_field("U", H1, base, 3);

    // Add field representing ho-geometry
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, base, 3);

    // Add entities to field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "U");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBVERTEX, "U", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "U", order);
    CHKERR m_field.set_field_order(0, MBTRI, "U", order);
    CHKERR m_field.set_field_order(0, MBTET, "U", order);

    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
    // CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);

    // setup the modules
    boost::ptr_vector<GenericElementInterface> m_modules;

    // Basic Boundary Conditions module should always be first (dirichlet)
    m_modules.push_back(new BasicBoundaryConditionsInterface(
        m_field, "U", "MESH_NODE_POSITIONS", simple->getProblemName(),
        simple->getDomainFEName(), true, is_quasi_static, nullptr,
        is_partitioned));

    m_modules.push_back(new NonlinearElasticElementInterface(
        m_field, "U", "MESH_NODE_POSITIONS", true, is_quasi_static));

    m_modules.push_back(
        new MFrontMoFEMInterface(m_field, "U", "MESH_NODE_POSITIONS", true, is_quasi_static));

    for (auto &&mod : m_modules) {
      mod.getCommandLineParameters();
      mod.addElementFields();
    }

    // build fields
    CHKERR m_field.build_fields();
    for (auto &&mod : m_modules)
      mod.createElements();

    Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);

    CHKERR m_field.build_finite_elements();
    CHKERR m_field.build_adjacencies(simple->getBitRefLevel());

    DMType dm_name = "DMMOFEM";
    // Register DM problem
    CHKERR DMRegister_MoFEM(dm_name);
    dM = createSmartDM(m_field.get_comm(), dm_name);
    CHKERR DMSetType(dM, dm_name);
    CHKERR DMMoFEMSetIsPartitioned(dM, is_partitioned);
    // Create DM instance
    CHKERR DMMoFEMCreateMoFEM(dM, &m_field, simple->getProblemName().c_str(),
                              simple->getBitRefLevel());
    CHKERR DMSetFromOptions(dM);

    for (auto &&mod : m_modules) {
      CHKERR mod.setOperators();
      CHKERR mod.addElementsToDM(dM);
    }

    CHKERR DMSetUp(dM);
    monitor_ptr = boost::make_shared<FEMethod>();
    monitor_ptr->preProcessHook = []() { return 0; };
    monitor_ptr->operatorHook = []() { return 0; };
    monitor_ptr->postProcessHook = [&]() {
      MoFEMFunctionBegin;
      // auto ts_time = monitor_ptr->ts_t;
      auto ts_step = monitor_ptr->ts_step;

      for (auto &&mod : m_modules) {
        CHKERR mod.updateElementVariables();
        if (ts_step % save_every_nth_step == 0)
          CHKERR mod.postProcessElement(ts_step);
      }

      MoFEMFunctionReturn(0);
    };

    auto t_type = GenericElementInterface::IM2;
    if (is_quasi_static)
      t_type = GenericElementInterface::IM;
    for (auto &&mod : m_modules) {
      CHKERR mod.setupSolverFunctionTS(t_type);
      CHKERR mod.setupSolverJacobianTS(t_type);
    }

    auto set_time_monitor = [&](auto solver) {
      MoFEMFunctionBegin;
      boost::shared_ptr<ForcesAndSourcesCore> null;
      CHKERR DMMoFEMTSSetMonitor(dM, solver, simple->getDomainFEName(),
                                 monitor_ptr, null, null);
      MoFEMFunctionReturn(0);
    };

    auto set_dm_section = [&](auto dm) {
      MoFEMFunctionBeginHot;
      auto section = m_field.getInterface<ISManager>()->sectionCreate(
          simple->getProblemName());
      CHKERR DMSetSection(dm, section);
      MoFEMFunctionReturnHot(0);
    };

    CHKERR set_dm_section(dM);

    auto D = smartCreateDMVector(dM);
    tSolver = MoFEM::createTS(m_field.get_comm());

    if (is_quasi_static) {
      CHKERR TSSetSolution(tSolver, D);
    } else {
      CHKERR TSSetType(tSolver, TSALPHA2);
      auto DD = smartVectorDuplicate(D);
      CHKERR TS2SetSolution(tSolver, D, DD);
    }

    CHKERR TSSetDM(tSolver, dM);

    // default max time is 1
    CHKERR TSSetMaxTime(tSolver, 1.0);
    CHKERR TSSetExactFinalTime(tSolver, TS_EXACTFINALTIME_MATCHSTEP);
    CHKERR TSSetFromOptions(tSolver);

    CHKERR set_time_monitor(tSolver);

    CHKERR TSSetUp(tSolver);
    CHKERR TSSolve(tSolver, NULL);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}