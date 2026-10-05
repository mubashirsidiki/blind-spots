import os
import sys
import json
import time
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        return f"Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB"
    return "CPU (CUDA not available)"


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def main():
    overall_start = time.perf_counter()

    print("=" * 70)
    print("  Qwen3-VL-2B Urdu Nastaliq Menu Evaluation (Step-by-Step with Timers)")
    print("=" * 70)

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 1. Resolve local model path
    local_model_dir = os.path.join(PROJECT_ROOT, "models", "Qwen3-VL-2B-Instruct")
    if os.path.exists(local_model_dir) and os.path.exists(
        os.path.join(local_model_dir, "model.safetensors")
    ):
        model_id = local_model_dir
        print(
            f"[Model Source] Using local self-contained model folder:\n  -> {local_model_dir}"
        )
    else:
        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        print(
            f"[Model Source] Local model folder not found, falling back to HuggingFace Hub: {model_id}"
        )

    print(
        f"[GPU Status] Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"[GPU Memory] Initial: {get_gpu_memory()}\n")

    # 2. Timing: Processor Loading
    print("--- [Timer: Step 1/2] Loading AutoProcessor & Tokenizer ---")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    t_proc = time.perf_counter() - t0
    print(f"-> Processor loaded in {format_duration(t_proc)}\n")

    # 3. Timing: Model Weights Loading
    print("--- [Timer: Step 2/2] Loading Model Weights into GPU VRAM (bfloat16) ---")
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    t_model = time.perf_counter() - t0
    print(f"-> Model loaded in {format_duration(t_model)}")
    print(f"-> GPU Memory after load: {get_gpu_memory()}\n")

    images_dir = os.path.join(PROJECT_ROOT, "dataset", "menu")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    image_files = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
    evaluation_records = []

    # Aggregated time tracking stats
    timing_stats = {
        "model_loading_seconds": t_proc + t_model,
        "preprocessing_seconds": 0.0,
        "generation_seconds": 0.0,
        "decoding_seconds": 0.0,
        "total_generated_tokens": 0,
        "query_runs": [],
    }

    print("=" * 70)
    print("Starting Menu Inference Runs")
    print("=" * 70)

    for img_idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found. Skipping.")
            continue

        print(
            "\n======================================================================"
        )
        print(f"  [Image {img_idx}/{len(image_files)}] {img_file}")
        print("======================================================================")

        t0_img = time.perf_counter()
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        t_img_open = time.perf_counter() - t0_img
        print(f"Image opened: {w}x{h} px in {format_duration(t_img_open)}")

        if img_file == "1.jpg":
            prompts = [
                (
                    "full_transcription",
                    "Read all text from this restaurant cover page, including English and Urdu, exactly as written.",
                )
            ]
        else:
            prompts = [
                (
                    "english_extraction",
                    "Extract all English dish names and their prices from this menu. Format as a clean list: 'Dish Name - Price'.",
                ),
                (
                    "urdu_extraction",
                    "Extract all Urdu dish names and their prices from this menu. Format as a clean list: 'اردو نام - قیمت'.",
                ),
                (
                    "bilingual_alignment",
                    "For each item in this menu, extract the English dish name, the Urdu dish name, and the price in PKR. Format each item on a new line.",
                ),
            ]

        page_record = {"image": img_file, "dimensions": f"{w}x{h}", "queries": {}}

        for test_idx, (test_name, prompt_text) in enumerate(prompts, 1):
            query_start = time.perf_counter()
            print(
                "\n  ------------------------------------------------------------------"
            )
            print(f"  [Test {test_idx}/{len(prompts)}] {test_name}")
            print(f'  Prompt: "{prompt_text}"')
            print(
                "  ------------------------------------------------------------------"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # 1. Preprocessing & Chat Template Tokenization
            t_pre_start = time.perf_counter()
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            t_pre = time.perf_counter() - t_pre_start
            num_input_tokens = inputs.input_ids.shape[1]
            timing_stats["preprocessing_seconds"] += t_pre

            print(
                f"  [Time Log] Step 1: Pre-process & Tokenize : {format_duration(t_pre)} (Input tokens: {num_input_tokens})"
            )

            # 2. Generation Forward Pass
            t_gen_start = time.perf_counter()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False
                )
            t_gen = time.perf_counter() - t_gen_start
            timing_stats["generation_seconds"] += t_gen

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            num_output_tokens = len(generated_ids_trimmed[0])
            timing_stats["total_generated_tokens"] += num_output_tokens
            tokens_per_sec = (num_output_tokens / t_gen) if t_gen > 0 else 0

            print(
                f"  [Time Log] Step 2: Model Generation Pass: {format_duration(t_gen)} ({num_output_tokens} tokens @ {tokens_per_sec:.1f} tok/s)"
            )

            # 3. Decoding
            t_dec_start = time.perf_counter()
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            t_dec = time.perf_counter() - t_dec_start
            timing_stats["decoding_seconds"] += t_dec

            total_query_time = time.perf_counter() - query_start
            print(
                f"  [Time Log] Step 3: Token Decoding       : {format_duration(t_dec)}"
            )
            print(
                f"  [Time Log] Total Query Latency         : {format_duration(total_query_time)}"
            )
            print(f"  [VRAM] {get_gpu_memory()}")

            snippet = output_text.strip()
            print(f"\n  --- Preview of Generated Output ({len(snippet)} chars) ---")
            lines = snippet.split("\n")
            preview_lines = lines[:6]
            for pl in preview_lines:
                print(f"    {pl}")
            if len(lines) > 6:
                print(f"    ... [{len(lines) - 6} more lines]")
            print("  ------------------------------------------------------------")

            query_timing = {
                "test_name": test_name,
                "input_tokens": num_input_tokens,
                "output_tokens": num_output_tokens,
                "preprocessing_time_sec": round(t_pre, 3),
                "generation_time_sec": round(t_gen, 3),
                "decoding_time_sec": round(t_dec, 3),
                "total_time_sec": round(total_query_time, 3),
                "tokens_per_second": round(tokens_per_sec, 1),
            }

            timing_stats["query_runs"].append(query_timing)

            page_record["queries"][test_name] = {
                "prompt": prompt_text,
                "timing": query_timing,
                "response": output_text,
            }

        evaluation_records.append(page_record)

    total_eval_duration = time.perf_counter() - overall_start

    # Save detailed JSON output
    out_json_path = os.path.join(results_dir, "qwen3_vl_eval_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timing_summary": {
                    "total_duration_seconds": round(total_eval_duration, 2),
                    "model_loading_seconds": round(
                        timing_stats["model_loading_seconds"], 2
                    ),
                    "preprocessing_seconds": round(
                        timing_stats["preprocessing_seconds"], 2
                    ),
                    "generation_seconds": round(timing_stats["generation_seconds"], 2),
                    "decoding_seconds": round(timing_stats["decoding_seconds"], 2),
                    "total_generated_tokens": timing_stats["total_generated_tokens"],
                    "avg_generation_speed_tok_s": (
                        round(
                            timing_stats["total_generated_tokens"]
                            / timing_stats["generation_seconds"],
                            1,
                        )
                        if timing_stats["generation_seconds"] > 0
                        else 0
                    ),
                },
                "pages": evaluation_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Generate Markdown Summary with Timing Section
    summary_md_path = os.path.join(results_dir, "evaluation_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Qwen3-VL-2B-Instruct Evaluation on Pakistani Urdu Menus\n\n")
        f.write(
            "- **Model**: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)\n"
        )
        f.write(f"- **Evaluated Images**: {len(image_files)}\n")
        f.write(
            f"- **Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
        )
        f.write(f"- **Total Runtime**: {format_duration(total_eval_duration)}\n\n")

        f.write("## Performance & Timing Breakdown\n\n")
        f.write("| Operation | Total Time | Percentage of Total Runtime |\n")
        f.write("|---|---|---|\n")
        m_load = timing_stats["model_loading_seconds"]
        m_pre = timing_stats["preprocessing_seconds"]
        m_gen = timing_stats["generation_seconds"]
        m_dec = timing_stats["decoding_seconds"]
        f.write(
            f"| Model & Processor Loading | {format_duration(m_load)} | {m_load / total_eval_duration * 100:.1f}% |\n"
        )
        f.write(
            f"| Image Preprocessing & Tokenization | {format_duration(m_pre)} | {m_pre / total_eval_duration * 100:.1f}% |\n"
        )
        f.write(
            f"| Model Generation Forward Pass | {format_duration(m_gen)} | {m_gen / total_eval_duration * 100:.1f}% |\n"
        )
        f.write(
            f"| Output Decoding | {format_duration(m_dec)} | {m_dec / total_eval_duration * 100:.1f}% |\n\n"
        )
        f.write(
            f"- **Total Tokens Generated**: {timing_stats['total_generated_tokens']}\n"
        )
        if m_gen > 0:
            f.write(
                f"- **Average Generation Speed**: {timing_stats['total_generated_tokens'] / m_gen:.1f} tokens/second\n\n"
            )

        for page in evaluation_records:
            f.write(f"## Page: `{page['image']}`\n\n")
            for q_type, q_data in page["queries"].items():
                t = q_data["timing"]
                f.write(f"### Test: `{q_type}`\n")
                f.write(f"- **Prompt**: {q_data['prompt']}\n")
                f.write(
                    f"- **Latency**: {t['total_time_sec']}s (Gen: {t['generation_time_sec']}s, Speed: {t['tokens_per_second']} tok/s, Tokens: {t['output_tokens']})\n\n"
                )
                f.write("**Model Output**:\n```\n")
                f.write(q_data["response"].strip())
                f.write("\n```\n\n")

    print("\n" + "=" * 70)
    print("                    TIMING BREAKDOWN SUMMARY")
    print("=" * 70)
    print(f"Total Runtime:                      {format_duration(total_eval_duration)}")
    print(
        f"  1. Model Loading Time:            {format_duration(m_load)} ({m_load / total_eval_duration * 100:.1f}%)"
    )
    print(
        f"  2. Image Preprocessing & Tokens:  {format_duration(m_pre)} ({m_pre / total_eval_duration * 100:.1f}%)"
    )
    print(
        f"  3. Generation Forward Pass (GPU): {format_duration(m_gen)} ({m_gen / total_eval_duration * 100:.1f}%)"
    )
    print(
        f"  4. Text Token Decoding:           {format_duration(m_dec)} ({m_dec / total_eval_duration * 100:.1f}%)"
    )
    if m_gen > 0:
        print(
            f"\nThroughput: {timing_stats['total_generated_tokens']} tokens generated at {timing_stats['total_generated_tokens'] / m_gen:.1f} tokens/second"
        )
    print("=" * 70)
    print(f"Results saved to: {out_json_path}")
    print(f"Summary markdown saved to: {summary_md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
