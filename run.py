import pandas as pd
import sys

from tqdm import tqdm
from pathlib import Path
from utils import is_better, suppress_noisy_logs, GREEN, RESET

sys.path.insert(0, "methods/ttsds/src")

import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
    
suppress_noisy_logs()


if __name__ == "__main__":


    folder = "data/gozde"
    multi_input_folder_flag = True
    metric_list = [
        "audiobox_pq",
        "brouhaha",
        "dnsmos_pro",
        "dnsmos_pro_bvcc",
        "dnsmos_pro_nisqa",
        "dnsmos_pro_vcc2018",
        "speaker_similarity",
        "scoreq_synthetic_nr",
        # "scoreq_synthetic_ref",
        "low_level_dc_offset",
        "low_level_lufs",
        "low_level_peak_clips",
        "low_level_sibilance_ratio",
        "squim_pesq",
        "squim_stoi",
        "squim_si_sdr",
        # "squim_mos",
        "ttsds",
        "utmos_v2",
         "whisqa",
        # "natscore",
        "speechlmscore",
    ]

    metric_list = ["audiobox_pq", "brouhaha", "dnsmos_pro", "utmos_v2", "whisqa"]
    save_path = "results_" + Path(folder).name + ".csv"

    # =============================================================
    
    assert len(metric_list) > 0, "At least one metric must be specified"
    assert Path(folder).exists(), f"Input folder {folder} does not exist"

    input_files = {}
    if multi_input_folder_flag:
        for file in Path(folder).glob("**/*.wav"):
            if file.parent.name not in input_files:
                input_files[file.parent.name] = []
            input_files[file.parent.name].append(str(file))
    else:
        input_files[Path(folder).name] = [str(file) for file in Path(folder).glob("*.wav")]

    results = {}
    
    # init metrics funcs and store in dict
    metric_funcs = {}
    low_level = None
    squim = None
    scoreq_model = None
    dnsmos_pro_model = None
    natscore_model = None
    speechlmscore_model = None
    speaker_similarity_refs = {}
    for metric in tqdm(metric_list, desc="Initializing metrics"):
        if metric == "audiobox_pq":
            from metrics.audiobox_pq import AudioboxPQ
            metric_funcs[metric] = AudioboxPQ()
        elif metric == "brouhaha":
            from metrics.brouhaha import Brouhaha
            metric_funcs[metric] = Brouhaha()
        elif metric == "speaker_similarity":
            from metrics.speaker_similarity import SpeakerSimilarity

            metric_funcs[metric] = SpeakerSimilarity()
            ref_root = Path(speaker_similarity_reference_folder)
            for input_name in input_files:
                ref_files = list((ref_root / input_name).glob("*.wav"))
                if len(ref_files) != 1:
                    raise ValueError(
                        f"Expected exactly one reference wav in {(ref_root / input_name)}, found {len(ref_files)}"
                    )
                speaker_similarity_refs[input_name] = str(ref_files[0])
        elif metric.startswith("low_level_"):
            if low_level is None:
                from metrics.low_level import LowLevel

                low_level = LowLevel()
            name = metric.replace("low_level_", "")
            metric_funcs[metric] = lambda audio_path, n=name: low_level(audio_path, name=n)
        elif metric.startswith("squim_"):
            if squim is None:
                from metrics.squim import Squim

                squim = Squim()
            name = metric.replace("squim_", "")
            metric_funcs[metric] = lambda audio_path, n=name: squim(audio_path, name=n)
        elif metric.startswith("scoreq_"):
            if scoreq_model is None:
                from metrics.scoreq_metric import ScoreQ

                scoreq_model = ScoreQ()
            name = metric.replace("scoreq_", "")
            metric_funcs[metric] = lambda audio_path, reference_path=None, n=name: scoreq_model(
                audio_path,
                name=n,
                reference_path=reference_path,
            )
            if name.endswith("_ref"):
                ref_root = Path(speaker_similarity_reference_folder)
                for input_name in input_files:
                    ref_files = list((ref_root / input_name).glob("*.wav"))
                    if len(ref_files) != 1:
                        raise ValueError(
                            f"Expected exactly one reference wav in {(ref_root / input_name)}, found {len(ref_files)}"
                        )
                    speaker_similarity_refs[input_name] = str(ref_files[0])
        elif metric == "ttsds":
            from metrics.ttsds_metric import TTSDSMetric
            ttsds_suite_save_path = None
            if save_path:
                ttsds_suite_save_path = save_path.replace(".csv", "_ttsds_detailed.csv")

            metric_funcs[metric] = TTSDSMetric(
                reference_root=speaker_similarity_reference_folder,
                write_to_file=ttsds_suite_save_path,
            )
        elif metric == "utmos_v2":
            from metrics.utmos_v2 import UTMOSv2
            metric_funcs[metric] = UTMOSv2()
        elif metric == "whisqa":
            from metrics.whisqa import WhiSQA

            metric_funcs[metric] = WhiSQA(
                model_type="single",
                checkpoints_dir="methods/WhiSQA/checkpoints",
            )
        elif metric == "natscore":
            if natscore_model is None:
                from metrics.natscore_metric import NatScoreMetric

                natscore_model = NatScoreMetric()
            metric_funcs[metric] = natscore_model
        elif metric == "dnsmos_pro" or metric.startswith("dnsmos_pro_"):
            if dnsmos_pro_model is None:
                from metrics.dnsmos_pro import DNSMOSPro
                dnsmos_pro_model = DNSMOSPro()
            if metric == "dnsmos_pro":
                metric_funcs[metric] = lambda audio_path: dnsmos_pro_model(audio_path, name="nisqa")
            else:
                _dnsmos_name = metric.replace("dnsmos_pro_", "")
                metric_funcs[metric] = lambda audio_path, n=_dnsmos_name: dnsmos_pro_model(audio_path, name=n)
        elif metric == "speechlmscore":
            if speechlmscore_model is None:
                from metrics.speechlmscore import SpeechLMScore
                speechlmscore_model = SpeechLMScore()
            metric_funcs[metric] = speechlmscore_model
        else:
            raise ValueError(f"Metric {metric} not recognized")

    result_row_list = []

    def print_progress_tables(completed_metrics):
        import statistics
        from tabulate import tabulate

        if not completed_metrics:
            return

        best_by_metric_avg = {}
        best_by_metric_med = {}

        for metric_name in completed_metrics:
            metric_inputs = results.get(metric_name, {})
            for input_name, input_results in metric_inputs.items():
                avg_result = sum(input_results) / len(input_results) if input_results else 0
                med_result = statistics.median(input_results) if input_results else 0
                if metric_name not in best_by_metric_avg or is_better(metric_name, avg_result, best_by_metric_avg[metric_name][1]):
                    best_by_metric_avg[metric_name] = (input_name, avg_result)
                if metric_name not in best_by_metric_med or is_better(metric_name, med_result, best_by_metric_med[metric_name][1]):
                    best_by_metric_med[metric_name] = (input_name, med_result)

        avg_func = lambda xs: sum(xs) / len(xs) if xs else 0
        med_func = lambda xs: statistics.median(xs) if xs else 0

        def build_rows(agg_func, best_by_metric):
            rows = []
            for input_name in input_files:
                row = [input_name]
                for metric_name in completed_metrics:
                    metric_values = results.get(metric_name, {}).get(input_name, [])
                    val = agg_func(metric_values)
                    if metric_name in best_by_metric and input_name == best_by_metric[metric_name][0]:
                        row.append(f"{GREEN}{val:.4f}{RESET}")
                    else:
                        row.append(f"{val:.4f}")
                rows.append(row)
            return rows

        print("\n--- Mean (Progress) ---")
        print(tabulate(build_rows(avg_func, best_by_metric_avg), headers=["Input", *completed_metrics]))
        print("\n--- Median (Progress) ---")
        print(tabulate(build_rows(med_func, best_by_metric_med), headers=["Input", *completed_metrics]))
    
    # run each metric on each input file and store the results in the results dictionary
    completed_metrics = []
    for metric in metric_list:
        metric_func = metric_funcs[metric]

        if metric == "ttsds":
            if metric not in results:
                results[metric] = {}

            ttsds_results = metric_func.process_all(input_files)
            for input_name, metric_scores in tqdm(ttsds_results.items(), desc=f"Running metric {metric}"):
                overall_score = float(metric_scores.get("overall", 0.0)) if isinstance(metric_scores, dict) else float(metric_scores)
                results[metric][input_name] = [overall_score]

                if isinstance(metric_scores, dict):
                    for sub_metric in sorted(metric_scores.keys()):
                        if sub_metric == "overall":
                            continue
                        result_row_list.append({
                            "Metric": f"ttsds_{sub_metric}",
                            "Input": input_name,
                            "File": input_name,
                            "Result": metric_scores[sub_metric],
                        })

                result_row_list.append({
                    "Metric": "ttsds",
                    "Input": input_name,
                    "File": input_name,
                    "Result": overall_score,
                })

            completed_metrics.append(metric)
            print_progress_tables(completed_metrics)
            continue

        for input_name, audio_path_list in tqdm(input_files.items(), desc=f"Running metric {metric}"):

            for input_file in audio_path_list:
                if metric not in results:
                    results[metric] = {}
                if input_name not in results[metric]:
                    results[metric][input_name] = []

                if metric == "speaker_similarity" or metric.endswith("_ref"):
                    result = metric_func(input_file, speaker_similarity_refs[input_name])
                else:
                    result = metric_func(input_file)
                results[metric][input_name].append(result)
                result_row_list.append({
                    "Metric": metric,
                    "Input": input_name,
                    "File": input_file,
                    "Result": result,
                })

        completed_metrics.append(metric)
        print_progress_tables(completed_metrics)

    # print average results for each metric and input as table with tabulate
    import statistics
    from tabulate import tabulate
    long_table = []
    best_by_metric_avg = {}
    best_by_metric_med = {}
    for metric, inputs in results.items():
        for input_name, input_results in inputs.items():
            avg_result = sum(input_results) / len(input_results) if input_results else 0
            med_result = statistics.median(input_results) if input_results else 0
            long_table.append([metric, input_name, avg_result])
            if metric not in best_by_metric_avg or is_better(metric, avg_result, best_by_metric_avg[metric][1]):
                best_by_metric_avg[metric] = (input_name, avg_result)
            if metric not in best_by_metric_med or is_better(metric, med_result, best_by_metric_med[metric][1]):
                best_by_metric_med[metric] = (input_name, med_result)

    def build_wide_rows(agg_func, best_by_metric):
        rows = []
        for input_name in input_files:
            row = [input_name]
            for metric in metric_list:
                val = agg_func(results[metric][input_name])
                if input_name == best_by_metric[metric][0]:
                    row.append(f"{GREEN}{val:.4f}{RESET}")
                else:
                    row.append(f"{val:.4f}")
            rows.append(row)
        return rows

    avg_func = lambda xs: sum(xs) / len(xs) if xs else 0
    med_func = lambda xs: statistics.median(xs) if xs else 0

    print("\n--- Mean ---")
    print(tabulate(build_wide_rows(avg_func, best_by_metric_avg), headers=["Input", *metric_list]))
    print("\n--- Median ---")
    print(tabulate(build_wide_rows(med_func, best_by_metric_med), headers=["Input", *metric_list]))


    if save_path:
        # save to csv using pandas
        df = pd.DataFrame(long_table, columns=["Metric", "Input", "Average Result"])
        df.to_csv(save_path.replace(".csv", "_average.csv"), index=False)
        
        # save detailed results to csv using pandas
        detailed_df = pd.DataFrame(result_row_list)
        # detailed_df.to_csv(save_path.replace(".csv", "_detailed.csv"), index=False)
        wide_df = detailed_df.pivot_table(
            index=["Input", "File"],
            columns="Metric",
            values="Result",
            aggfunc="first"
        ).reset_index()

        wide_df.to_csv(save_path.replace(".csv", "_detailed.csv"), index=False)

   
