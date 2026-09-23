"""
Comprehensive Stress Test Suite (45+ Real-World Scenarios)
Classifying Text Documents Using Machine Learning

Evaluates model behavior across:
- Pure single-topic examples (Space, Baseball, Politics, Graphics)
- Short sentences ('NASA launch', 'baseball game')
- Long multi-sentence articles
- Mixed-topic sentences ('The president discussed NASA's Mars program')
- Unrelated / Out-of-Domain text ('I ate pizza today', 'My laptop battery is low')
- Ambiguous multi-topic queries ('Chief Minister went to Mars to play football')
- Specific required test: 'The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.'
- Noisy text (punctuation, casing, URLs, numbers, emojis)
- Questions, statements, technical, and casual language
- Malformed, empty, and whitespace-only inputs
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference_engine import (
    classify_document,
    InferenceConfig,
    TARGET_4_CLASSES,
    CATEGORY_DISPLAY_NAMES_4
)

STRESS_TEST_CASES = [
    # -------------------------------------------------------------------------
    # 1. Pure Single-Topic Examples
    # -------------------------------------------------------------------------
    {
        "id": "pure_space_1",
        "category": "pure_space",
        "text": "NASA launched a spacecraft into orbit using advanced cryogenic rocket propulsion.",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "pure_space_2",
        "category": "pure_space",
        "text": "The Hubble space telescope captured distant galaxies and interstellar nebulae.",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "pure_baseball_1",
        "category": "pure_baseball",
        "text": "The baseball team scored five runs in the bottom of the ninth inning to win.",
        "expected_type": "in_domain",
        "target_class": "rec.sport.baseball"
    },
    {
        "id": "pure_baseball_2",
        "category": "pure_baseball",
        "text": "The starting pitcher recorded twelve strikeouts and allowed zero walks in the game.",
        "expected_type": "in_domain",
        "target_class": "rec.sport.baseball"
    },
    {
        "id": "pure_politics_1",
        "category": "pure_politics",
        "text": "The government passed a new law regarding federal taxation and civil liberties.",
        "expected_type": "in_domain",
        "target_class": "talk.politics.misc"
    },
    {
        "id": "pure_politics_2",
        "category": "pure_politics",
        "text": "Congressional representatives held a heated debate over constitutional amendments and voting rights.",
        "expected_type": "in_domain",
        "target_class": "talk.politics.misc"
    },
    {
        "id": "pure_graphics_1",
        "category": "pure_graphics",
        "text": "The computer generated a 3D image using polygon mesh shading and ray tracing.",
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },
    {
        "id": "pure_graphics_2",
        "category": "pure_graphics",
        "text": "Real-time rasterization and shader compilation optimize GPU rendering pipelines.",
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },

    # -------------------------------------------------------------------------
    # 2. Short Sentences
    # -------------------------------------------------------------------------
    {
        "id": "short_space",
        "category": "short_text",
        "text": "NASA launch",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "short_baseball",
        "category": "short_text",
        "text": "baseball game",
        "expected_type": "in_domain",
        "target_class": "rec.sport.baseball"
    },
    {
        "id": "short_politics",
        "category": "short_text",
        "text": "new government law",
        "expected_type": "in_domain",
        "target_class": "talk.politics.misc"
    },
    {
        "id": "short_graphics",
        "category": "short_text",
        "text": "3D rendering software",
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },

    # -------------------------------------------------------------------------
    # 3. Long Multi-Sentence Documents
    # -------------------------------------------------------------------------
    {
        "id": "long_space",
        "category": "long_document",
        "text": (
            "The European Space Agency and NASA have jointly developed the next generation "
            "deep space exploration orbiter. Equipped with solar electric propulsion, "
            "the planetary probe will traverse the asteroid belt and enter elliptical orbit "
            "around Jupiter. Scientific telemetry instruments will analyze atmospheric composition, "
            "magnetic fields, and potential subsurface oceans on Europa and Ganymede."
        ),
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "long_graphics",
        "category": "long_document",
        "text": (
            "Modern graphics processing units leverage specialized tensor and ray-tracing cores "
            "to perform real-time volumetric rendering and path tracing. The rendering pipeline "
            "processes complex polygon geometry through vertex, geometry, and fragment shaders. "
            "Spatial acceleration structures like bounding volume hierarchies significantly reduce "
            "ray intersection computations for realistic lighting, ambient occlusion, and reflections."
        ),
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },

    # -------------------------------------------------------------------------
    # 4. Mixed-Topic Sentences (Multi-domain overlaps)
    # -------------------------------------------------------------------------
    {
        "id": "mixed_govt_mars",
        "category": "mixed_topic",
        "text": "The government announced a new Mars mission.",
        "expected_type": "mixed_or_ambiguous"
    },
    {
        "id": "mixed_baseball_nasa",
        "category": "mixed_topic",
        "text": "The baseball team visited NASA.",
        "expected_type": "mixed_or_ambiguous"
    },
    {
        "id": "mixed_president_football",
        "category": "mixed_topic",
        "text": "The president attended a football match.",
        "expected_type": "mixed_or_ambiguous"
    },
    {
        "id": "mixed_astronaut_graphics",
        "category": "mixed_topic",
        "text": "The astronaut used a computer graphics system.",
        "expected_type": "mixed_or_ambiguous"
    },
    {
        "id": "mixed_president_mars",
        "category": "mixed_topic",
        "text": "The president discussed NASA's Mars program.",
        "expected_type": "mixed_or_ambiguous"
    },
    {
        "id": "mixed_astronaut_baseball",
        "category": "mixed_topic",
        "text": "The astronaut watched a baseball game.",
        "expected_type": "mixed_or_ambiguous"
    },

    # -------------------------------------------------------------------------
    # 5. The Benchmark Multi-Topic Stress Case
    # -------------------------------------------------------------------------
    {
        "id": "ambiguous_chief_minister",
        "category": "ambiguous_stress",
        "text": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.",
        "expected_type": "ambiguous_or_distributed"
    },
    {
        "id": "ambiguous_cm_mars_football",
        "category": "ambiguous_stress",
        "text": "Chief Minister went to Mars to play football.",
        "expected_type": "ambiguous_or_distributed"
    },

    # -------------------------------------------------------------------------
    # 6. Completely Unrelated / Out-of-Domain Text
    # -------------------------------------------------------------------------
    {
        "id": "ood_pizza",
        "category": "out_of_domain",
        "text": "I ate pizza today.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_battery",
        "category": "out_of_domain",
        "text": "My laptop battery is low.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_shopping",
        "category": "out_of_domain",
        "text": "I went shopping yesterday.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_python",
        "category": "out_of_domain",
        "text": "Python is my favorite programming language.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_movie",
        "category": "out_of_domain",
        "text": "The movie was amazing.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_motorcycle",
        "category": "out_of_domain",
        "text": "Royal Enfield launched a new motorcycle.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_phone",
        "category": "out_of_domain",
        "text": "I bought a new phone yesterday.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_bike_starting",
        "category": "out_of_domain",
        "text": "My Royal Enfield is not starting.",
        "expected_type": "unknown"
    },
    {
        "id": "ood_sports_random",
        "category": "out_of_domain",
        "text": "football cricket basketball tennis",
        "expected_type": "unknown_or_low_conf"
    },
    {
        "id": "ood_recipe",
        "category": "out_of_domain",
        "text": "Mix flour, sugar, butter, and two eggs in a bowl to bake delicious cookies.",
        "expected_type": "unknown"
    },

    # -------------------------------------------------------------------------
    # 7. Noisy Text (Punctuation, URLs, casing, repeated words, emojis)
    # -------------------------------------------------------------------------
    {
        "id": "noisy_space_url",
        "category": "noisy_text",
        "text": "NASA launched a probe!!! Check it out at https://nasa.gov/mission?id=99283 #SpaceExploration 🚀🚀",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "noisy_baseball_caps",
        "category": "noisy_text",
        "text": "THE PITCHER STRUCK OUT 10 BATTERS IN INNING 9!!! BASEBALL CHAMPIONSHIP WON!!!",
        "expected_type": "in_domain",
        "target_class": "rec.sport.baseball"
    },
    {
        "id": "noisy_graphics_mixed",
        "category": "noisy_text",
        "text": "3D 3D 3D render render!! email questions to support@3dgraphics.org or call 1-800-555-0199.",
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },

    # -------------------------------------------------------------------------
    # 8. Questions and Casual Inquiries
    # -------------------------------------------------------------------------
    {
        "id": "question_space",
        "category": "questions",
        "text": "Can NASA launch the spacecraft tomorrow?",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "question_baseball",
        "category": "questions",
        "text": "Who is pitching in tonight's baseball game?",
        "expected_type": "in_domain",
        "target_class": "rec.sport.baseball"
    },
    {
        "id": "question_politics",
        "category": "questions",
        "text": "Will Congress pass the new government bill this week?",
        "expected_type": "in_domain",
        "target_class": "talk.politics.misc"
    },
    {
        "id": "casual_graphics",
        "category": "casual_hardware",
        "text": "My graphics card is overheating.",
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },

    # -------------------------------------------------------------------------
    # 9. Edge Cases: Empty, Very Short, Malformed
    # -------------------------------------------------------------------------
    {
        "id": "edge_empty",
        "category": "edge_case",
        "text": "",
        "expected_type": "unknown"
    },
    {
        "id": "edge_whitespace",
        "category": "edge_case",
        "text": "   \n\t   ",
        "expected_type": "unknown"
    },
    {
        "id": "edge_single_char",
        "category": "edge_case",
        "text": "a",
        "expected_type": "unknown"
    },
    {
        "id": "edge_punctuation_only",
        "category": "edge_case",
        "text": "!@#$%^&*()_+=-{}[]:;'<>?,./",
        "expected_type": "unknown"
    },
    {
        "id": "edge_numbers_only",
        "category": "edge_case",
        "text": "129847192837 91283719283",
        "expected_type": "unknown"
    },

    # -------------------------------------------------------------------------
    # 10. Technical Statements & Additional Real-World Scenarios
    # -------------------------------------------------------------------------
    {
        "id": "stmt_space_microgravity",
        "category": "pure_space",
        "text": "The astronaut conducted research on microgravity inside the International Space Station.",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "stmt_baseball_grand_slam",
        "category": "pure_baseball",
        "text": "The batter hit a grand slam into deep center field to tie the baseball game.",
        "expected_type": "in_domain",
        "target_class": "rec.sport.baseball"
    },
    {
        "id": "stmt_politics_judiciary",
        "category": "pure_politics",
        "text": "The Senate judiciary committee reviewed presidential executive orders and constitutional law.",
        "expected_type": "in_domain",
        "target_class": "talk.politics.misc"
    },
    {
        "id": "stmt_graphics_shaders",
        "category": "pure_graphics",
        "text": "OpenGL and Vulkan enable parallel GPU compute shaders for 3D ray tracing and mesh rendering.",
        "expected_type": "in_domain",
        "target_class": "comp.graphics"
    },
    {
        "id": "technical_telemetry_space",
        "category": "pure_space",
        "text": "Orbital telemetry transmissions confirmed that the interplanetary probe entered Martian orbit successfully.",
        "expected_type": "in_domain",
        "target_class": "sci.space"
    },
    {
        "id": "casual_groceries_dinner",
        "category": "out_of_domain",
        "text": "I need to buy groceries and cook dinner tonight.",
        "expected_type": "unknown"
    }
]


def run_stress_test_suite():
    print("=" * 85)
    print(" EXECUTING DEDICATED STRESS TEST SUITE (45+ REAL-WORLD SCENARIOS)")
    print("=" * 85)

    # Load artifacts
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(best_model_path) or not os.path.exists(vec_path):
        print("[ERROR] Model artifacts not found. Please train models first using 'python train.py'.")
        sys.exit(1)

    model = joblib.load(best_model_path)
    vectorizer = joblib.load(vec_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    categories = meta.get("categories", TARGET_4_CLASSES)
    config = InferenceConfig()

    results = []
    category_summary = {}

    for case in STRESS_TEST_CASES:
        c_id = case["id"]
        c_group = case["category"]
        raw_text = case["text"]
        expected_type = case["expected_type"]

        res = classify_document(raw_text, model, vectorizer, categories=categories, config=config)

        # Check status and properties
        status = res["status"]
        pred = res["prediction"]
        conf = res["confidence"]
        detected = res["detected_topics"]

        category_summary.setdefault(c_group, {"total": 0, "normal": 0, "ambiguous": 0, "unknown": 0})
        category_summary[c_group]["total"] += 1
        category_summary[c_group][status] += 1

        results.append({
            "id": c_id,
            "group": c_group,
            "text": raw_text[:50] + ("..." if len(raw_text) > 50 else ""),
            "status": status,
            "prediction": pred,
            "confidence": f"{conf * 100:.1f}%",
            "detected_topics": ", ".join(detected[:3])
        })

    df_res = pd.DataFrame(results)

    print("\n" + "-" * 85)
    print(" SAMPLE STRESS TEST RESULTS (FIRST 25 CASES):")
    print("-" * 85)
    print(df_res.head(25).to_string(index=False))

    print("\n" + "-" * 85)
    print(" SUMMARY BY TEST GROUP:")
    print("-" * 85)
    for grp, counts in category_summary.items():
        print(f"  • {grp.upper():<20} Total: {counts['total']:<3} | Normal: {counts['normal']:<3} | Ambiguous: {counts['ambiguous']:<3} | Unknown: {counts['unknown']:<3}")

    # Specific Verification for required benchmark input:
    target_prompt = "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
    print("\n" + "=" * 85)
    print(f" BENCHMARK TEST EVALUATION:")
    print(f" Input: \"{target_prompt}\"")
    print("=" * 85)
    target_res = classify_document(target_prompt, model, vectorizer, categories=categories, config=config)
    print(f"  Status:          {target_res['status']}")
    print(f"  Prediction:      {target_res['prediction']}")
    print(f"  Confidence:      {target_res['confidence'] * 100:.2f}%")
    print(f"  Detected Topics: {target_res['detected_topics']}")
    print(f"  Probabilities:")
    for cat, p in target_res['probabilities'].items():
        disp = CATEGORY_DISPLAY_NAMES_4.get(cat, cat)
        print(f"    - {disp:<20} ({cat}): {p * 100:.2f}%")
    print(f"  Top Keywords:    {[kw['term'] for kw in target_res['top_keywords']]}")
    print(f"  Reason:          {target_res['reason']}")
    print("=" * 85)

    # Verification assertions
    assert len(results) >= 40, f"Expected >= 40 stress tests, got {len(results)}"

    # Out of domain tests should be classified as Unknown
    ood_cases = [r for r in results if r["group"] == "out_of_domain"]
    unknown_ood_count = sum(1 for r in ood_cases if r["status"] == "unknown")
    print(f"\n[Verification] Out-of-Domain Unknown Detection Rate: {unknown_ood_count}/{len(ood_cases)} ({unknown_ood_count/len(ood_cases):.1%})")

    # Edge cases (empty, whitespace) should all be Unknown
    edge_cases = [r for r in results if r["group"] == "edge_case"]
    unknown_edge_count = sum(1 for r in edge_cases if r["status"] == "unknown")
    assert unknown_edge_count == len(edge_cases), "All empty/malformed edge cases must be classified as Unknown"

    print("\n[SUCCESS] Dedicated Stress Test Suite completed successfully!")
    return df_res, target_res


if __name__ == "__main__":
    run_stress_test_suite()
