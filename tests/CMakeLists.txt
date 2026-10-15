# =============================================================================
# gsolver Tests
# =============================================================================

include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------
add_executable(gsolver_tests
    test_boundary_edges.cpp
    test_factor_graph.cpp
    test_se2_factors.cpp
    test_se2_gbp.cpp
)
target_link_libraries(gsolver_tests PRIVATE gsolver::gsolver GTest::gtest_main)
# One OpenMP thread per test: the graphs are tiny, and ctest runs tests in
# parallel, so a full thread team per process would oversubscribe the cores
gtest_discover_tests(gsolver_tests PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

# -----------------------------------------------------------------------------
# Compile-failure tests: each source must be rejected by a static_assert in
# FactorGraph::addFactorNode, with the expected message.
# -----------------------------------------------------------------------------
set(COMPILE_FAIL_CASES
    "typed_wrong_type|variable types don't match the factor's variable types"
    "typed_wrong_count|wrong number of variables for this factor"
)
foreach(case_spec ${COMPILE_FAIL_CASES})
    string(REPLACE "|" ";" case_parts "${case_spec}")
    list(GET case_parts 0 case_name)
    list(GET case_parts 1 case_message)

    add_executable(compile_fail_${case_name} EXCLUDE_FROM_ALL compile_fail/${case_name}.cpp)
    target_link_libraries(compile_fail_${case_name} PRIVATE gsolver::gsolver)
    add_test(
        NAME compile_fail.${case_name}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target compile_fail_${case_name}
    )
    # Pass only if the build output contains the expected static_assert message
    set_tests_properties(compile_fail.${case_name} PROPERTIES PASS_REGULAR_EXPRESSION "${case_message}")
endforeach()
