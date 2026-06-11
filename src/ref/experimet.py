import os
import argparse
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import onnx
import si4onnx
from si4onnx.operators import Abs, AverageFilter, InputDiff, L1Norm
from environment import set_env
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import (
    BrainDataset,
    MVtecDataset,
    generate_images_non_iid,
    generate_images_corr,
    generate_images_iid,
)


def fn(args):
    input_x, ref_x, mask, category, type, thr, image_size, var, seed, idx, n_jobs = args
    synthetic_catgory = {
        "iid",
        "corr",
        "skewnorm",
        "exponnorm",
        "gennormsteep",
        "gennormflat",
        "t",
    }

    if n_jobs == 1:
        set_env(parallel=False)
        torch.set_num_threads(1)
    else:
        set_env(parallel=True)

    input_x = input_x.clone().detach().to(dtype=torch.float64)
    ref_x = ref_x.clone().detach().to(dtype=torch.float64)

    match category:
        case "no_tumor" | "tumor":
            timesteps = 300
            step = 75
            onnx_path = (
                "../model/brain/"
                + f"sim_diffusion_size{image_size}"
                + f"_timesteps{timesteps}"
                + f"_step{step}.onnx"
            )
        case (
            "bottle"
            | "cable"
            | "carpet"
            | "grid"
            | "leather"
            | "metal_nut"
            | "tile"
            | "transistor"
            | "wood"
            | "pill"
            | "screw"
        ):
            timesteps = 500
            step = 125
            onnx_path = (
                f"../model/MVtec/{category}/"
                + f"sim_diffusion_size{image_size}"
                + f"_timesteps{timesteps}"
                + f"_step{step}.onnx"
            )
        case _:
            timesteps = 460
            step = 115
            onnx_path = (
                "../model/syn/"
                + f"sim_diffusion_size{image_size}"
                + f"_timesteps{timesteps}"
                + f"_step{step}.onnx"
            )

    onnx_model = onnx.load(onnx_path)

    match category:
        case "screw":
            si_model = si4onnx.load(
                model=onnx_model,
                hypothesis=si4onnx.NeighborMeanDiff(
                    threshold=thr,
                    neighborhood_range=12,
                    post_process=[InputDiff(), AverageFilter(), L1Norm()],
                ),
                seed=seed,
            )
        case _:
            si_model = si4onnx.load(
                model=onnx_model,
                hypothesis=si4onnx.ReferenceMeanDiff(
                    threshold=thr,
                    post_process=[InputDiff(), AverageFilter(), L1Norm()],
                ),
                seed=seed,
            )
            

    try:
        match category:
            case "screw":
                oc_result = si_model.inference(
                    input=input_x,
                    var=var,
                    mask=mask,
                    inference_mode="over_conditioning",
                )
                start_time = time.time()
                pp_result = si_model.inference(
                    input=input_x,
                    var=var,
                    mask=mask,
                    inference_mode="parametric",
                    n_jobs=n_jobs,
                    max_iter=2e7,
                )
            case _:
                oc_result = si_model.inference(
                    (input_x, ref_x),
                    var=var,
                    mask=mask,
                    inference_mode="over_conditioning",
                )
                start_time = time.time()
                pp_result = si_model.inference(
                    (input_x, ref_x),
                    var=var,
                    mask=mask,
                    inference_mode="parametric",
                    n_jobs=n_jobs,
                    max_iter=2e7,
                )
        calc_time = time.time() - start_time
        # selective_p = oc_result.p_value # デバッグ用：テスト時はコメントアウト
        selective_p = pp_result.p_value
        oc_p = oc_result.p_value
        naive_p = oc_result.naive_p_value()
        z = oc_result.stat

        output = si_model.output
        salient_region = si_model.roi
        score_map = si_model.score_map

        # Compute permutation p-value
        # perm_rng = np.random.default_rng(seed)
        # permutation_p_value = None
        # if category in synthetic_catgory:
        #     corr_z_list = []
        #     B = 1000
        #     cnt = 0
        #     permutation_error = 0
        #     while cnt < B:
        #         if permutation_error > 1000:
        #             return None
        #         try:
        #             x_permutated = torch.cat([input_x, ref_x], dim=0)
        #             x_permutated = (
        #                 x_permutated.view(-1)[
        #                     perm_rng.permutation(x_permutated.numel())
        #                 ]
        #                 .reshape(x_permutated.shape)
        #                 .double()
        #             )

        #             ref_x_permutated = x_permutated[:1]
        #             input_x_permutated = x_permutated[1:]

        #             si_model.hypothesis.reference_data = ref_x_permutated
        #             si_model.construct_hypothesis(input_x_permutated)
        #             permutation_z = si_model.si_calculator.stat
        #             corr_z_list.append(np.abs(permutation_z))
        #             cnt += 1
        #         except:
        #             permutation_error += 1
        #             continue
        #     permutation_p_value = 1 / B * np.sum(np.array(corr_z_list) > np.abs(z))
        permutation_p_value = -1.0
    except:
        print(None)
        traceback.print_exc()
        return None

    if category not in synthetic_catgory:
        result = {
            "category": category,
            "image_size": image_size,
            "seed": seed,
            "selective_p_value": selective_p,
            "oc_p_value": oc_p,
            "naive_p_value": naive_p,
            "z": z,
            "input_image": input_x,
            "output_image": output,
            "salient_region": salient_region,
            "score_map": score_map,
            "time": calc_time,
            "reference_image": ref_x,
        }

        if selective_p is not None:
            save_dir = f"../results/{category}/thr{thr}/{type}/"
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, f"id_{idx}.pickle"), "wb") as f:
                pickle.dump(result, f)

    return (
        selective_p,
        oc_p,
        naive_p,
        z,
        output,
        salient_region,
        ref_x,
        permutation_p_value,
        calc_time,
    )


def experiment(
    category: str,
    type: str,
    image_size: int,
    thr: float,
    signal: float,
    seed: int,
    number_of_workers: int,
    num_iter: int,
    n_jobs: int = 1,
    distance: float = None,
    **kwargs,
):
    print(
        f"category: {category}",
        f"type: {type}",
        f"image_size: {image_size}",
        f"signal: {signal}",
        f"seed: {seed}",
    )
    mask_list = []
    match category:
        case "iid":
            input_image_list, _, _ = generate_images_iid(
                num=num_iter,
                img_size=image_size,
                scale=1,
                signal=signal,
                seed=seed,
            )
            reference_image_list, _, _ = generate_images_iid(
                num=num_iter,
                img_size=image_size,
                scale=1,
                signal=0,
                seed=int(seed + 2023),
            )
            input_image_list = torch.from_numpy(input_image_list).to(torch.float64)
            reference_image_list = torch.from_numpy(reference_image_list).to(
                torch.float64
            )
            mask_list = [None] * num_iter
            var = 1.0

        case "corr":
            input_image_list, cov = generate_images_corr(
                num=num_iter, img_size=image_size, signal=signal, seed=seed
            )
            reference_image_list, cov = generate_images_corr(
                num=num_iter, img_size=image_size, signal=0, seed=int(seed + 2023)
            )
            input_image_list = torch.from_numpy(input_image_list).to(torch.float64)
            reference_image_list = torch.from_numpy(reference_image_list).to(
                torch.float64
            )
            ZERO = np.zeros((image_size**2, image_size**2))
            top = np.concatenate([cov, ZERO], axis=1)
            bottom = np.concatenate([ZERO, cov], axis=1)
            var = np.concatenate([top, bottom], axis=0)
            mask_list = [None] * num_iter

        case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
            image_list = generate_images_non_iid(
                image_size, category, distance, num_samples=num_iter * 2
            )
            input_image_list = image_list[:num_iter]
            reference_image_list = image_list[num_iter:]
            input_image_list = torch.from_numpy(input_image_list).to(torch.float64)
            reference_image_list = torch.from_numpy(reference_image_list).to(
                torch.float64
            )
            var = 1.0
            mask_list = [None] * num_iter

        case "no_tumor" | "tumor":
            input_image_list = []
            if category == "no_tumor":
                dataset = BrainDataset(mode="test", type="normal")
            elif category == "tumor":
                dataset = BrainDataset(mode="test", type="abnormal")
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            for image, mask, _ in data_loader:
                input_image_list.append(image[0])
                mask_list.append(mask[0])
            input_image_list = torch.stack(input_image_list, dim=0)

            reference_image_list = []
            dataset = BrainDataset(mode="test", type="reference")
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for image, _, _ in data_loader:
                reference_image_list.append(image[0])

            reference_image_list = torch.stack(reference_image_list, dim=0)
            var = 1.0

        case (
            "bottle"
            | "cable"
            | "carpet"
            | "grid"
            | "leather"
            | "metal_nut"
            | "tile"
            | "transistor"
            | "wood"
            | "pill"
            | "screw"
        ):
            # test image
            input_image_list = []
            dataset = MVtecDataset(
                category=category,
                mode="test",
                type=type,
                img_size=image_size,
            )
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for image, mask, seg in data_loader:
                input_image_list.append(image[0])
                if type == "normal":
                    mask_list.append(None)
                elif type == "abnormal":
                    mask_list.append(seg[0])

            input_image_list = torch.stack(input_image_list, dim=0)

            # reference image
            reference_image_list = []
            dataset = MVtecDataset(
                category=category,
                mode="test",
                type="reference",
                img_size=image_size,
            )

            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for image, _, _ in data_loader:
                reference_image_list.append(image[0])

            reference_image_list = torch.stack(reference_image_list, dim=0)

            var = 1.0

    seeds = np.arange(num_iter)

    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        args = (
            (
                input_image_list[idx : idx + 1, :, :, :],
                reference_image_list[idx : idx + 1, :, :, :],
                mask_list[idx],
                category,
                type,
                thr,
                image_size,
                var,
                seeds[idx],
                idx,
                n_jobs,
            )
            for idx in range(num_iter)
        )
        outputs = list(tqdm(executor.map(fn, args), total=num_iter))

        p_values = []
        input_images = []
        output_images = []
        permutation_p_values = []
        salient_regions = []
        reference_images = []
        times = []
        for i in range(num_iter):
            if (
                outputs[i] is None
                or None in outputs[i][0:4]  # selective, oc, naive, z
                or outputs[i][5] is None  # salient_region
            ):
                continue
            p_values.append(outputs[i][0:4])
            input_images.append(input_image_list[i : i + 1, :, :, :])
            output_images.append(outputs[i][4])
            salient_regions.append(outputs[i][5])
            reference_images.append(outputs[i][6])
            permutation_p_values.append(outputs[i][7])
            times.append(outputs[i][8])

        result = np.array([p_value for p_value in p_values if p_value is not None])
        selective_p_values = result[:, 0]
        oc_p_values = result[:, 1]
        naive_p_values = result[:, 2]
        z = result[:, 3]

    print("naive:", len(naive_p_values))

    result_dict = {
        "category": category,
        "image_size": image_size,
        "signal": signal,
        "num_iter": num_iter,
        "seed": seed,
        "selective_p_values": selective_p_values,
        "oc_p_values": oc_p_values,
        "naive_p_values": naive_p_values,
        "permutation_p_values": permutation_p_values,
        "z": z,
        "time": times,
    }

    if category in ["no_tumor", "tumor"]:
        result_dict["input_images"] = input_images
        result_dict["salient_regions"] = salient_regions
        result_dict["output_images"] = output_images
        result_dict["reference_images"] = reference_images

    if signal == 0:
        error = "fpr"
    else:
        error = "power"

    match category:
        case "iid":
            with open(
                f"../results/iid/{error}/"
                + f"iid_size{image_size}"
                + f"_signal{int(signal)}"
                + f"_seed{seed}"
                + ".pickle",
                "wb",
            ) as f:
                pickle.dump(result_dict, f)
        case "corr":
            with open(
                f"../results/corr/{error}/"
                + f"corr_size{image_size}"
                + f"_signal{int(signal)}"
                + f"_seed{seed}"
                + ".pickle",
                "wb",
            ) as f:
                pickle.dump(result_dict, f)
        case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
            with open(
                f"../results/robust/{error}/"
                + f"{category}_size{image_size}"
                + f"_distance{distance}"
                + f"_seed{seed}"
                + ".pickle",
                "wb",
            ) as f:
                pickle.dump(result_dict, f)
        case "no_tumor" | "tumor":
            with open(
                "../results/thr0.65/brain/"
                + f"{category}"
                + f"_seed{seed}"
                + ".pickle",
                "wb",
            ) as f:
                pickle.dump(result_dict, f)
        case (
            "bottle"
            | "cable"
            | "carpet"
            | "grid"
            | "leather"
            | "metal_nut"
            | "tile"
            | "transistor"
            | "wood"
            | "pill"
        ):
            save_dir = f"../results/{category}/thr{thr}/"
            os.makedirs(save_dir, exist_ok=True)
            file_name = (
                f"{category}_{type}" + f"_size{image_size}" + f"_seed{seed}" + ".pickle"
            )
            with open(os.path.join(save_dir, file_name), "wb") as f:
                pickle.dump(result_dict, f)


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("-category", "--category", type=str)
    cmdline_parser.add_argument("-type", "--type", type=str)
    cmdline_parser.add_argument("-size", "--size", type=int)
    cmdline_parser.add_argument("-thr", "--thr", type=float)
    cmdline_parser.add_argument("-signal", "--signal", type=float)
    cmdline_parser.add_argument("-workers", "--workers", type=int)
    cmdline_parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1)
    cmdline_parser.add_argument("-iter", "--iter", type=int)
    cmdline_parser.add_argument("-seed", "--seed", type=int)
    cmdline_parser.add_argument("-distance", "--distance", type=float, default=None)

    args, unknowns = cmdline_parser.parse_known_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    experiment(
        category=args.category,
        type=args.type,
        image_size=args.size,
        thr=args.thr,
        signal=args.signal,
        seed=args.seed,
        number_of_workers=args.workers,
        n_jobs=args.n_jobs,
        num_iter=args.iter,
        distance=args.distance,
    )
