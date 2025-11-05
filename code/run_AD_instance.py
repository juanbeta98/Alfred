from src.run_baselines import run_online_hist_baseline, run_online_algo_baseline, run_online_algo_baseline_parallel
from src.run_ONLINE_static import run_ONLINE_static
from src.run_INSERT import run_INSERT
from src.run_REACT import run_REACT

import argparse

def main():
    # ====== MANUAL DEBUG CONFIGURATION ======
    DEBUG_MODE = True  # ⬅️ Set to False when running from the command line
    DEFAULT_INSTANCE = "instAD2b"

    # ====== Run configuration ======
    optimization_obj = 'driver_distance'         # Options: ['hybrid', 'driver_distance', 'driver_extra_time']
    distance_method = 'haversine'       # Options: ['precalced', 'haversine']
    save_results = True

    # ====== CLI PARSING ======
    parser = argparse.ArgumentParser(description="Run online dynamic insertion simulation")
    parser.add_argument("--inst", help="Instance name (e.g. instAD2b)")
    args = parser.parse_args()

    # ====== INSTANCE SELECTION ======
    if DEBUG_MODE:
        instance = DEFAULT_INSTANCE
        print(f"[DEBUG MODE] Using default instance: {instance}")
    else:
        if not args.inst:
            parser.error("You must provide --instance when not in debug mode.")
        # instance = args.instance
        instance = f'inst{args.inst}'

    # ====== EXECUTION ======
    print(f"\n========================= Running online orchestration for {instance} =========================")
    print(f'--------- Optimization objective: {optimization_obj}')
    print(f'--------- Distance method: {distance_method}')
    print(f'--------- Save results: {save_results}\n')

    # run_online_hist_baseline(
    #     instance,
    #     distance_method=distance_method,
    #     save_results=save_results
    # )
    
    # run_online_algo_baseline(
    #     instance,
    #     optimization_obj=optimization_obj,
    #     distance_method=distance_method,
    #     save_results=save_results)
    
    # run_online_algo_baseline_parallel(
    #     instance,
    #     optimization_obj=optimization_obj,
    #     distance_method=distance_method,
    #     save_results=save_results)
    
    # run_ONLINE_static(
    #     instance,
    #     optimization_obj=optimization_obj,
    #     distance_method=distance_method)
    
    # run_INSERT(
    #     instance,
    #     optimization_obj=optimization_obj,
    #     distance_method=distance_method
    # )
    
    run_REACT(
        instance,
        optimization_obj=optimization_obj,
        distance_method=distance_method,
        time_previous_freeze=30,
        save_results=save_results
    )

    print(f"\n✅ Full pipeline for {instance} completed successfully.\n")


if __name__ == "__main__":
    main()
