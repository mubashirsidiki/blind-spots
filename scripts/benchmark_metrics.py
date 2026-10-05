import os
import sys
import json
import re
from typing import List, Tuple, Dict, Any, Optional

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def levenshtein_distance(seq1: str, seq2: str) -> int:
    """Standard dynamic programming Levenshtein edit distance."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    return dp[m][n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance(ref, hyp) / len(ref)."""
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = levenshtein_distance(ref_clean, hyp_clean)
    return dist / len(ref_clean)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: word_level_edit_distance(ref_words, hyp_words) / len(ref_words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / len(ref_words)


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation without stripping leading brand digits."""
    if not text:
        return ""
    # Strip leading bullets, hyphens, asterisks
    t = re.sub(r"^[\s\-\*\•]+", "", text).strip()
    # Strip list item numbering (e.g., '1.', '1)', '1 - ') only if followed by separator
    t = re.sub(r"^\d+[\.\)]\s*|^\d+\s*-\s+", "", t).strip()
    # Normalize multiple whitespaces
    t = re.sub(r"\s+", " ", t)
    return t


def parse_extracted_lines(response_text: str) -> List[Tuple[str, str]]:
    """Parse 'Name - Price' or 'Name Price' lines from model response."""
    items = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip list markers
        clean_line = re.sub(r"^[\s\-\*\•]+", "", line).strip()
        clean_line = re.sub(r"^\d+[\.\)]\s*|^\d+\s*-\s+", "", clean_line).strip()
        if not clean_line:
            continue

        # Look for price separator (hyphen, comma, slash, or colon)
        match = re.search(r"^(.*?)\s*[-:,]\s*([\d\s/]+)$", clean_line)
        if match:
            dish = normalize_text(match.group(1))
            price = match.group(2).strip()
            items.append((dish, price))
        else:
            # Fallback: check trailing numbers
            match_num = re.search(r"^(.*?)\s+(\d+(?:\s*/\s*\d+)?)$", clean_line)
            if match_num:
                dish = normalize_text(match_num.group(1))
                price = match_num.group(2).strip()
                items.append((dish, price))
            else:
                items.append((normalize_text(clean_line), ""))
    return items


def get_gt_price_str(gt_item: Dict[str, Any]) -> str:
    """Extract standard string representation of ground truth price."""
    if "price" in gt_item and gt_item["price"]:
        return str(gt_item["price"]).strip()
    if "price_full" in gt_item and "price_half" in gt_item:
        return f"{gt_item['price_full']} / {gt_item['price_half']}"
    return ""


def check_price_match(gt_item: Dict[str, Any], pred_price: str) -> Optional[bool]:
    """Check if model predicted price matches ground truth. Returns None if item has no price."""
    gt_p = get_gt_price_str(gt_item)
    if not gt_p:
        return None
    if not pred_price:
        return False
    pred_clean = re.sub(r"\s+", "", pred_price)
    if "price" in gt_item and gt_item["price"]:
        target = str(gt_item["price"]).strip()
        return target in pred_clean
    if "price_full" in gt_item and "price_half" in gt_item:
        pf = str(gt_item["price_full"]).strip()
        ph = str(gt_item["price_half"]).strip()
        return (pf in pred_clean) and (ph in pred_clean)
    return False


def align_items(
    gt_items: List[Dict[str, Any]],
    pred_items: List[Tuple[str, str]],
    lang_key: str = "urdu",
) -> List[Tuple[Dict[str, Any], Tuple[str, str]]]:
    """
    Monotonic dynamic programming alignment between ground truth items and predicted lines.
    Prevents single inserted lines (such as category headers) from cascading alignment errors.
    """
    n_gt = len(gt_items)
    n_pred = len(pred_items)

    if n_gt == n_pred:
        return list(zip(gt_items, pred_items))

    def match_cost(i_idx: int, j_idx: int) -> float:
        gt = gt_items[i_idx]
        pred_dish, pred_p = pred_items[j_idx]
        gt_target = normalize_text(gt.get(lang_key) or "")

        # Price matching provides a strong structural anchor
        p_match = check_price_match(gt, pred_p)
        if p_match is True:
            p_cost = 0.0
        elif p_match is False:
            p_cost = 3.0
        else:
            p_cost = 0.5

        # Character error rate cost
        cer = compute_cer(gt_target.lower(), pred_dish.lower())
        # Cross-script fallback (in case model transliterated English into Arabic script)
        gt_en = normalize_text(gt.get("english") or "")
        cer_en = compute_cer(gt_en.lower(), pred_dish.lower())
        return p_cost + min(cer, cer_en)

    dp = [[1e9] * (n_pred + 1) for _ in range(n_gt + 1)]
    parent: List[
        List[
            Optional[Tuple[int, int, Optional[Tuple[Dict[str, Any], Tuple[str, str]]]]]
        ]
    ] = [[None] * (n_pred + 1) for _ in range(n_gt + 1)]
    dp[0][0] = 0.0

    for j in range(1, n_pred + 1):
        dp[0][j] = dp[0][j - 1] + 1.0  # Insertion cost
        parent[0][j] = (0, j - 1, None)

    for i in range(1, n_gt + 1):
        dp[i][0] = dp[i - 1][0] + 5.0  # Deletion cost
        parent[i][0] = (i - 1, 0, (gt_items[i - 1], ("", "")))

    for i in range(1, n_gt + 1):
        for j in range(1, n_pred + 1):
            cost_m = dp[i - 1][j - 1] + match_cost(i - 1, j - 1)
            cost_skip_pred = dp[i][j - 1] + 1.0
            cost_skip_gt = dp[i - 1][j] + 5.0

            best_cost = cost_m
            best_p: Tuple[
                int, int, Optional[Tuple[Dict[str, Any], Tuple[str, str]]]
            ] = (i - 1, j - 1, (gt_items[i - 1], pred_items[j - 1]))

            if cost_skip_pred < best_cost:
                best_cost = cost_skip_pred
                best_p = (i, j - 1, None)
            if cost_skip_gt < best_cost:
                best_cost = cost_skip_gt
                best_p = (i - 1, j, (gt_items[i - 1], ("", "")))

            dp[i][j] = best_cost
            parent[i][j] = best_p

    curr_i, curr_j = n_gt, n_pred
    aligned: List[Tuple[Dict[str, Any], Tuple[str, str]]] = []
    while curr_i > 0 or curr_j > 0:
        p_info = parent[curr_i][curr_j]
        if p_info is None:
            break
        prev_i, prev_j, pair = p_info
        if pair is not None:
            aligned.append(pair)
        curr_i, curr_j = prev_i, prev_j

    aligned.reverse()
    return aligned


def evaluate_page(
    image_name: str,
    gt_items: List[Dict[str, Any]],
    model_queries: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate predictions for a single page with alignment and price verification."""
    en_results = []
    ur_results = []

    # Dynamic extraction for Page 1 (Cover / Notice)
    if image_name == "1.jpg" and "full_transcription" in model_queries:
        transcription = model_queries["full_transcription"]["response"]
        lines = [
            line.strip() for line in transcription.strip().split("\n") if line.strip()
        ]
        latin_lines = [
            line
            for line in lines
            if not re.search(r"[\u0600-\u06FF]", line) and not line.startswith("http")
        ]
        urdu_lines = [line for line in lines if re.search(r"[\u0600-\u06FF]", line)]

        policy_lines = [line for line in latin_lines if "NOTE" in line.upper()]
        title_lines = [line for line in latin_lines if "NOTE" not in line.upper()]

        pred_en_1_1 = " ".join(title_lines)
        pred_en_1_2 = policy_lines[0] if policy_lines else ""
        pred_ur_1_1 = urdu_lines[0] if urdu_lines else ""

        # Extract predictions for the annotated notice items
        for gt in gt_items:
            gt_id = gt["id"]
            gt_en = normalize_text(gt.get("english") or "")
            gt_ur = normalize_text(gt.get("urdu") or "")

            if gt_id == "1_1":
                pred_en_dish = pred_en_1_1
                pred_ur_dish = pred_ur_1_1
            elif gt_id == "1_2":
                pred_en_dish = pred_en_1_2
                pred_ur_dish = ""
            else:
                pred_en_dish = ""
                pred_ur_dish = ""

            if gt_en:
                cer_en = compute_cer(gt_en.lower(), pred_en_dish.lower())
                wer_en = compute_wer(gt_en.lower(), pred_en_dish.lower())
                em_en = gt_en.lower() == pred_en_dish.lower()
                en_results.append(
                    {
                        "id": gt_id,
                        "image": image_name,
                        "ground_truth": gt_en,
                        "predicted": pred_en_dish,
                        "cer": round(cer_en, 4),
                        "wer": round(wer_en, 4),
                        "exact_match": em_en,
                        "price_gt": "",
                        "price_pred": "",
                        "price_match": None,
                    }
                )

            if gt_ur:
                cer_ur = compute_cer(gt_ur.lower(), pred_ur_dish.lower())
                wer_ur = compute_wer(gt_ur.lower(), pred_ur_dish.lower())
                em_ur = gt_ur.lower() == pred_ur_dish.lower()
                ur_results.append(
                    {
                        "id": gt_id,
                        "image": image_name,
                        "ground_truth": gt_ur,
                        "predicted": pred_ur_dish,
                        "cer": round(cer_ur, 4),
                        "wer": round(wer_ur, 4),
                        "exact_match": em_ur,
                        "price_gt": "",
                        "price_pred": "",
                        "price_match": None,
                    }
                )

        return {"english": en_results, "urdu": ur_results}

    # Pages 2-5: Menu Dish Extractions
    en_extracted: List[Tuple[str, str]] = []
    if "english_extraction" in model_queries:
        en_extracted = parse_extracted_lines(
            model_queries["english_extraction"]["response"]
        )

    ur_extracted: List[Tuple[str, str]] = []
    if "urdu_extraction" in model_queries:
        ur_extracted = parse_extracted_lines(
            model_queries["urdu_extraction"]["response"]
        )

    en_aligned = align_items(gt_items, en_extracted, lang_key="english")
    ur_aligned = align_items(gt_items, ur_extracted, lang_key="urdu")

    for gt, (pred_en_dish, pred_en_price) in en_aligned:
        gt_en = normalize_text(gt.get("english") or "")
        if gt_en:
            cer_en = compute_cer(gt_en.lower(), pred_en_dish.lower())
            wer_en = compute_wer(gt_en.lower(), pred_en_dish.lower())
            em_en = gt_en.lower() == pred_en_dish.lower()
            gt_price = get_gt_price_str(gt)
            p_match = check_price_match(gt, pred_en_price)

            en_results.append(
                {
                    "id": gt["id"],
                    "image": image_name,
                    "ground_truth": gt_en,
                    "predicted": pred_en_dish,
                    "cer": round(cer_en, 4),
                    "wer": round(wer_en, 4),
                    "exact_match": em_en,
                    "price_gt": gt_price,
                    "price_pred": pred_en_price,
                    "price_match": p_match,
                }
            )

    for gt, (pred_ur_dish, pred_ur_price) in ur_aligned:
        gt_ur = normalize_text(gt.get("urdu") or "")
        if gt_ur:
            cer_ur = compute_cer(gt_ur.lower(), pred_ur_dish.lower())
            wer_ur = compute_wer(gt_ur.lower(), pred_ur_dish.lower())
            em_ur = gt_ur.lower() == pred_ur_dish.lower()
            gt_price = get_gt_price_str(gt)
            p_match = check_price_match(gt, pred_ur_price)

            ur_results.append(
                {
                    "id": gt["id"],
                    "image": image_name,
                    "ground_truth": gt_ur,
                    "predicted": pred_ur_dish,
                    "cer": round(cer_ur, 4),
                    "wer": round(wer_ur, 4),
                    "exact_match": em_ur,
                    "price_gt": gt_price,
                    "price_pred": pred_ur_price,
                    "price_match": p_match,
                }
            )

    return {"english": en_results, "urdu": ur_results}


def calculate_metrics_summary(
    en_results: List[Dict[str, Any]], ur_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate aggregated metrics for a given subset of results."""
    total_en = len(en_results)
    total_ur = len(ur_results)

    avg_en_cer = (sum(r["cer"] for r in en_results) / total_en) if total_en else 0.0
    avg_en_wer = (sum(r["wer"] for r in en_results) / total_en) if total_en else 0.0
    en_exact_acc = (
        (sum(1 for r in en_results if r["exact_match"]) / total_en * 100)
        if total_en
        else 0.0
    )

    # Price accuracy computed over items that actually have ground truth prices
    priced_items = [r for r in en_results if r.get("price_match") is not None]
    price_acc = (
        (sum(1 for r in priced_items if r["price_match"]) / len(priced_items) * 100)
        if priced_items
        else 0.0
    )

    avg_ur_cer = (sum(r["cer"] for r in ur_results) / total_ur) if total_ur else 0.0
    avg_ur_wer = (sum(r["wer"] for r in ur_results) / total_ur) if total_ur else 0.0
    ur_exact_acc = (
        (sum(1 for r in ur_results if r["exact_match"]) / total_ur * 100)
        if total_ur
        else 0.0
    )

    rounded_en_cer = round(avg_en_cer, 4)
    rounded_ur_cer = round(avg_ur_cer, 4)
    cer_ratio = (
        round(rounded_ur_cer / rounded_en_cer, 1) if rounded_en_cer > 0 else 999.0
    )

    rounded_en_wer = round(avg_en_wer, 4)
    rounded_ur_wer = round(avg_ur_wer, 4)
    wer_ratio = (
        round(rounded_ur_wer / rounded_en_wer, 1) if rounded_en_wer > 0 else 999.0
    )

    return {
        "items_count": {
            "english": total_en,
            "urdu": total_ur,
            "priced_items_checked": len(priced_items),
        },
        "english": {
            "character_error_rate_cer": rounded_en_cer,
            "word_error_rate_wer": rounded_en_wer,
            "exact_match_accuracy_pct": round(en_exact_acc, 2),
            "price_accuracy_pct": round(price_acc, 2),
        },
        "urdu_nastaliq": {
            "character_error_rate_cer": rounded_ur_cer,
            "word_error_rate_wer": rounded_ur_wer,
            "exact_match_accuracy_pct": round(ur_exact_acc, 2),
        },
        "disparity_ratio": {
            "cer_ratio_urdu_vs_english": cer_ratio,
            "wer_ratio_urdu_vs_english": wer_ratio,
            "accuracy_drop_pct": round(en_exact_acc - ur_exact_acc, 2),
        },
    }


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gt_path = os.path.join(PROJECT_ROOT, "dataset", "menu", "ground_truth.json")
    results_path = os.path.join(PROJECT_ROOT, "results", "qwen3_vl_eval_results.json")

    if not os.path.exists(gt_path) or not os.path.exists(results_path):
        print("Error: Required dataset or results file not found.")
        return

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(results_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    gt_by_page = {p["image"]: p["items"] for p in ground_truth["pages"]}
    pages_eval = {p["image"]: p["queries"] for p in eval_data["pages"]}

    all_en_results: List[Dict[str, Any]] = []
    all_ur_results: List[Dict[str, Any]] = []
    page_summaries: Dict[str, Any] = {}

    for img_file, gt_items in gt_by_page.items():
        if img_file not in pages_eval:
            continue
        res = evaluate_page(img_file, gt_items, pages_eval[img_file])
        all_en_results.extend(res["english"])
        all_ur_results.extend(res["urdu"])

        en_page = res["english"]
        ur_page = res["urdu"]
        page_summaries[img_file] = {
            "en_count": len(en_page),
            "ur_count": len(ur_page),
            "avg_en_cer": (
                round(sum(r["cer"] for r in en_page) / len(en_page), 4)
                if en_page
                else 0.0
            ),
            "avg_ur_cer": (
                round(sum(r["cer"] for r in ur_page) / len(ur_page), 4)
                if ur_page
                else 0.0
            ),
        }

    # Core menu items benchmark (pages 2-5, 58 items)
    core_en = [r for r in all_en_results if r["image"] != "1.jpg"]
    core_ur = [r for r in all_ur_results if r["image"] != "1.jpg"]
    core_metrics = calculate_metrics_summary(core_en, core_ur)

    # Full dataset benchmark (pages 1-5, 60 items)
    full_metrics = calculate_metrics_summary(all_en_results, all_ur_results)

    # Clean sample pairings by item id
    ur_by_id = {r["id"]: r for r in all_ur_results}
    sample_comparisons = []
    for r_en in all_en_results:
        item_id = r_en["id"]
        r_ur = ur_by_id.get(item_id)
        sample_comparisons.append(
            {
                "id": item_id,
                "image": r_en["image"],
                "ground_truth_en": r_en["ground_truth"],
                "predicted_en": r_en["predicted"],
                "ground_truth_ur": r_ur["ground_truth"] if r_ur else "",
                "predicted_ur": r_ur["predicted"] if r_ur else "",
                "cer_en": r_en["cer"],
                "cer_ur": r_ur["cer"] if r_ur else 0.0,
                "price_gt": r_en["price_gt"],
                "price_pred": r_en["price_pred"],
                "price_match": r_en["price_match"],
            }
        )

    report = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "benchmark_dataset": "Noorani Kabab House (Pakistani Urdu Menus)",
        "evaluation_scope": {
            "core_menu_dishes_pages_2_to_5": 58,
            "full_annotated_items_all_pages": 60,
        },
        "core_menu_dishes_metrics": core_metrics,
        "full_dataset_metrics": full_metrics,
        "page_breakdown": page_summaries,
        "sample_item_comparisons": sample_comparisons[:10],
    }

    out_file = os.path.join(PROJECT_ROOT, "results", "benchmark_scores.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("      QUANTITATIVE BENCHMARK: ENGLISH VS URDU NASTALIQ")
    cm_en = core_metrics["english"]
    cm_ur = core_metrics["urdu_nastaliq"]
    print("=" * 70)
    print(f"Model                      : {report['model']}")
    print(
        f"Core Menu Dishes Evaluated : {core_metrics['items_count']['english']} (Pages 2-5)"
    )
    print(
        f"Priced Items Verified      : {core_metrics['items_count']['priced_items_checked']}/{core_metrics['items_count']['english']} ({cm_en['price_accuracy_pct']:.1f}% match)"
    )
    print("-" * 70)
    print("Metric                       English (Latin)    Urdu (Nastaliq)")
    print("-" * 70)
    print(
        f"Exact Match Accuracy       :   {cm_en['exact_match_accuracy_pct']:>5.1f}%             {cm_ur['exact_match_accuracy_pct']:>5.1f}%"
    )
    print(
        f"Character Error Rate (CER) :   {cm_en['character_error_rate_cer']:>6.4f}             {cm_ur['character_error_rate_cer']:>6.4f}"
    )
    print(
        f"Word Error Rate (WER)      :   {cm_en['word_error_rate_wer']:>6.4f}             {cm_ur['word_error_rate_wer']:>6.4f}"
    )
    print(
        f"Price Accuracy             :   {cm_en['price_accuracy_pct']:>5.1f}%                 N/A"
    )
    print("-" * 70)
    disp = core_metrics["disparity_ratio"]
    print(
        f"Disparity Ratio: Urdu CER is {disp['cer_ratio_urdu_vs_english']}x higher than English."
    )
    print(
        f"Accuracy Drop  : {disp['accuracy_drop_pct']}% absolute drop on core menu dishes."
    )
    print("=" * 70)
    print(
        f"Saved detailed benchmark report to: [benchmark_scores.json](file:///{out_file})"
    )


if __name__ == "__main__":
    main()
