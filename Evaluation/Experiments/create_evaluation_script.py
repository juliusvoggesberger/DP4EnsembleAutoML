BLUEPRINT = "python run_auto_cen_dp.py {} {} {} {}\n"

budget_m = 100  # Configuration budget
seeds = [123, 456, 789, 1010, 2020]  # The random seed used in the evaluation
dc = ["SD", "HD", "CI"] # Data characteristics

dataset_ids_multi_dc = [40668, 1067, 40983, 1486, 4134, 40978]
dataset_ids = [1005, 40981, 1494, 23, 40984, 41145, 40926, 554, 40996, 41163, 40900, 30,
               1461] + dataset_ids_multi_dc

with open("commands.sh", "w") as f:
    f.write("#!/bin/bash \n")
    # Path to the virtual environment, if one exists
    f.write("source venv/bin/activate \n")
    for seed in seeds:
        for data_id in dataset_ids:
            f.write(BLUEPRINT.format(data_id, budget_m, seed, ""))

    for seed in seeds:
        for data_id in dataset_ids_multi_dc:
            for p in dc:
                f.write(BLUEPRINT.format(data_id, budget_m, seed, p))
