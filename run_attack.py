import argparse
import json
import multiprocessing as mp
import os
import time


DEFAULT_DATASET = "AdamCodd/Civitai-8m-prompts"
DEFAULT_TEXT_ENCODER = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_GENERATOR = "stabilityai/stable-diffusion-3.5-large-turbo"


def parse_args():
    parser = argparse.ArgumentParser(description="Run JANUS prompt attack experiments.")
    parser.add_argument(
        "-d",
        "--data",
        default=DEFAULT_DATASET,
        help="Prompt string, local JSON file, hf://datasets/<repo>/<file>, or Hugging Face dataset id.",
    )
    parser.add_argument(
        "-r",
        "--run-name",
        default="janus_run",
        help="Name of this run. Outputs are saved under --output-dir/<run-name>.",
    )
    parser.add_argument("-s", "--seed", required=True, type=int, help="Random seed.")
    parser.add_argument(
        "-n",
        "--num-candidates",
        type=int,
        default=5,
        help="Number of adversarial prompts sampled per input prompt.",
    )
    parser.add_argument(
        "-i",
        "--steps",
        type=int,
        default=5000,
        help="Prior optimization steps recorded for compatibility with paper runs.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts to load from a Hugging Face dataset. Use -1 for all local JSON rows.",
    )
    parser.add_argument(
        "--mask-sensitive-tokens",
        action="store_true",
        help="Mask obvious sensitive tokens during prompt sampling.",
    )
    parser.add_argument(
        "--save-prior",
        action="store_true",
        help="Save adapted JANUS prior parameters for each prompt.",
    )
    parser.add_argument("--output-dir", default="./outputs", help="Directory for run outputs.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split used for HF datasets.")
    parser.add_argument("--prompt-column", default="prompt", help="Prompt column used for HF datasets.")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER, help="SDXL text encoder repo id.")
    parser.add_argument("--generator", default=DEFAULT_GENERATOR, help="Image generator repo id.")
    parser.add_argument(
        "--t5-encoder",
        default=None,
        help="Optional T5 encoder repo id. Defaults to --generator/text_encoder_3.",
    )
    parser.add_argument(
        "--prior-repo",
        default="HaolunZheng/JANUS-priors",
        help="HF repo containing nmna_model1{index}.pth and nmna_model2{index}.pth.",
    )
    parser.add_argument(
        "--prior-dir",
        default="nmna_update_notau_dim1_all_mean_fuz_mask_absolute_data_RL_KL",
        help="Directory inside --prior-repo containing prior checkpoint files.",
    )
    parser.add_argument("--text-filter", default="michellejieli/NSFW_text_classifier")
    parser.add_argument("--image-scorer", default="Falconsai/nsfw_image_detection")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached HF files only.")
    parser.add_argument("--score-images", action="store_true", help="Enable slower image reward scoring.")
    parser.add_argument("--track", action="store_true", help="Enable SwanLab tracking.")
    return parser.parse_args()


args = parse_args()

import torch
from datasets import load_dataset
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download
from transformers import (
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
)

from utils.attack import JanusAttack


def resolve_hf_file(uri, token=None, local_files_only=False):
    if not uri.startswith("hf://"):
        return uri

    path = uri[len("hf://"):]
    repo_type = "model"
    if path.startswith("datasets/"):
        repo_type = "dataset"
        path = path[len("datasets/"):]
    elif path.startswith("models/"):
        path = path[len("models/"):]

    parts = path.split("/")
    if len(parts) < 3:
        raise ValueError("Expected hf://datasets/<namespace>/<repo>/<filename>.")

    return hf_hub_download(
        repo_id="/".join(parts[:2]),
        filename="/".join(parts[2:]),
        repo_type=repo_type,
        token=token,
        local_files_only=local_files_only,
    )


def looks_like_dataset_id(value):
    if value.startswith("hf://") or value.endswith(".json") or os.path.exists(value):
        return False
    return "/" in value and " " not in value


def load_prompts(args):
    source = resolve_hf_file(args.data, args.hf_token, args.local_files_only)
    if source.endswith(".json"):
        with open(source, "r") as f:
            data = json.load(f)
        prompts = data["prompts"]
        clean_prompts = data.get("clean_prompts", [""] * len(prompts))
        if args.num_prompts >= 0:
            prompts = prompts[: args.num_prompts]
            clean_prompts = clean_prompts[: args.num_prompts]
        return prompts, clean_prompts, data

    if looks_like_dataset_id(source):
        limit = args.num_prompts if args.num_prompts >= 0 else 200
        dataset = load_dataset(
            source,
            split=args.dataset_split,
            streaming=True,
            token=args.hf_token,
        )
        prompts = []
        for row in dataset:
            prompt = row.get(args.prompt_column)
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)
            if len(prompts) >= limit:
                break
        if not prompts:
            raise ValueError(f"No prompts loaded from {source}. Check --dataset-split and --prompt-column.")
        return prompts, [""] * len(prompts), {"prompts": prompts, "clean_prompts": [""] * len(prompts)}

    return [source], [""], None


def load_models(device, args):
    transformer_quant = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.generator,
        subfolder="transformer",
        quantization_config=transformer_quant,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        local_files_only=args.local_files_only,
    )

    t5_quant = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t5_kwargs = {} if args.t5_encoder else {"subfolder": "text_encoder_3"}
    t5 = T5EncoderModel.from_pretrained(
        args.t5_encoder or args.generator,
        quantization_config=t5_quant,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        local_files_only=args.local_files_only,
        **t5_kwargs,
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.generator,
        transformer=transformer,
        text_encoder_3=t5,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        local_files_only=args.local_files_only,
    ).to(device)

    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.text_encoder,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        local_files_only=args.local_files_only,
        use_safetensors=True,
    ).to(device=device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.text_encoder,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        local_files_only=args.local_files_only,
        use_safetensors=True,
    ).to(device=device)
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.text_encoder,
        subfolder="tokenizer",
        token=args.hf_token,
        local_files_only=args.local_files_only,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.text_encoder,
        subfolder="tokenizer_2",
        token=args.hf_token,
        local_files_only=args.local_files_only,
    )

    return pipe, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2


def run_on_gpu(device_idx, prompt_chunk, args, attack_config, output_dir, results):
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")
    print(f"GPU {device_idx}: attacking {len(prompt_chunk)} prompts.")

    pipe, model1, model2, tokenizer1, tokenizer2 = load_models(device, args)
    attack = JanusAttack(
        model=model1,
        model2=model2,
        tokenizer=tokenizer1,
        tokenizer2=tokenizer2,
        post="_mask" if args.mask_sensitive_tokens else "",
        output_dir=output_dir,
        seed=args.seed,
        device=device,
        pipe=pipe,
        prior_repo=args.prior_repo,
        prior_dir=args.prior_dir,
        text_filter=args.text_filter,
        image_scorer=args.image_scorer,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
        score_images=args.score_images,
        track=args.track,
    )

    for prompt_id, prompt, clean_prompt in prompt_chunk:
        attack.prompt = prompt
        attack.clean_prompt = clean_prompt
        start = time.time()
        if args.save_prior:
            candidates, prior1, prior2 = attack.attack(prompt_id=prompt_id, **attack_config)
            torch.save(prior1.state_dict(), os.path.join(output_dir, f"prior_text_{prompt_id}.pth"))
            torch.save(prior2.state_dict(), os.path.join(output_dir, f"prior_clean_{prompt_id}.pth"))
        else:
            candidates = attack.attack(prompt_id=prompt_id, **attack_config)

        results[f"target{prompt_id}_best_adv_prompts"] = candidates
        results[f"target{prompt_id}_runtime"] = time.time() - start
        print(f"GPU {device_idx}: finished prompt {prompt_id}.")

    del pipe, model1, model2, tokenizer1, tokenizer2, attack
    torch.cuda.empty_cache()


if __name__ == "__main__":
    prompts, clean_prompts, original_data = load_prompts(args)
    output_dir = os.path.abspath(os.path.join(args.output_dir, args.run_name))
    os.makedirs(output_dir, exist_ok=True)

    attack_config = {
        "steps": args.steps,
        "n_cand": args.num_candidates,
        "init": 1,
        "mask": int(args.mask_sensitive_tokens),
        "hard": 0,
        "save_model": int(args.save_prior),
    }

    full_prompt_list = list(zip(range(len(prompts)), prompts, clean_prompts))
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("JANUS requires at least one CUDA GPU.")

    chunk_size = (len(full_prompt_list) + num_gpus - 1) // num_gpus
    prompt_chunks = [
        full_prompt_list[i * chunk_size : min((i + 1) * chunk_size, len(full_prompt_list))]
        for i in range(num_gpus)
    ]
    prompt_chunks = [chunk for chunk in prompt_chunks if chunk]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_results = manager.dict()
    processes = []
    start_all = time.time()

    for gpu_id, chunk in enumerate(prompt_chunks):
        process = mp.Process(
            target=run_on_gpu,
            args=(gpu_id, chunk, args, attack_config, output_dir, shared_results),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_results = {}
    total_prompt_runtime = 0.0
    for key, value in shared_results.items():
        if key.endswith("_best_adv_prompts"):
            final_results[key] = value
        elif key.endswith("_runtime"):
            total_prompt_runtime += value

    final_results["runtime/iter"] = total_prompt_runtime / args.num_candidates / len(prompts)
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(final_results, f, indent=2)

    if original_data is not None:
        with open(os.path.join(output_dir, "prompts.json"), "w") as f:
            json.dump(original_data, f, indent=2)

    print(f"Finished in {time.time() - start_all:.2f}s. Results saved to {result_path}.")
