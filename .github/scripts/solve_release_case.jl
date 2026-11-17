using GenX
using JuMP

case_dir = abspath(only(ARGS))
settings_dir = joinpath(case_dir, "settings")
setup = GenX.configure_settings(
    joinpath(settings_dir, "genx_settings.yml"),
    joinpath(settings_dir, "output_settings.yml"),
)
# PowerGenome already reduced this example to 240 hours.
setup["TimeDomainReduction"] = 0
setup["MultiStage"] = 0
setup["Solver"] = "HiGHS"
optimizer = GenX.configure_solver(settings_dir, GenX.HiGHS.Optimizer; solver_name="highs")
inputs = GenX.load_inputs(setup, case_dir)
model = GenX.generate_model(setup, inputs, optimizer)
model, solve_time = GenX.solve_model(model, setup)
status = termination_status(model)
println("Release smoke test termination status: ", status)
status == MOI.OPTIMAL || error("GenX release smoke test did not solve to optimality: $status")
inputs["solve_time"] = solve_time
GenX.write_outputs(model, joinpath(case_dir, "results"), setup, inputs)
