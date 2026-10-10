"""
evaluate.py
===========
Evaluación de modelos entrenados con Label-Free RLVR.

Métricas implementadas:
  - Pass@1  : accuracy con decodificación greedy
  - Pass@K  : probabilidad de que al menos 1 de K muestras sea correcta
  - Cross-domain transfer: evaluar modelo en dominio no entrenado

Uso:
    # Evaluar modelo entrenado
    python evaluate.py --model_path runs/self_certainty__math__XXXXXX/final_model --domain math

    # Evaluar modelo base (baseline)
    python evaluate.py --model_path Qwen/Qwen2.5-1.5B --domain math --baseline

    # Evaluar transferencia cruzada
    python evaluate.py --model_path runs/self_certainty__math__XXXXXX/final_model --domain logic
    python evaluate.py --model_path runs/self_certainty__math__XXXXXX/final_model --domain code

Autor: Rhonald Jose Torres Diaz
Afiliación: Universidad Nacional de Colombia — Sede La Paz
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

EVAL_CONFIG = {
    "max_seq_length": 512,
    "load_in_4bit": True,
    "max_new_tokens": 256,
    "temperature": 0.8,       # Para muestreo en Pass@K
    "top_p": 0.95,
    "k_samples": 10,          # K para Pass@K
    "max_eval_samples": 200,  # Samples de evaluación por dominio
}

# Prompts de evaluación (mismos que train.py)
DOMAIN_PROMPTS = {
    "math": (
        "Solve the following math problem step by step.\n\n"
        "Problem: {question}\n\nSolution:"
    ),
    "logic": (
        "Answer the following multiple choice question.\n\n"
        "Question: {question}\n"
        "Choices: {choices}\n\nAnswer:"
    ),
    "code": (
        "Complete the following Python function.\n\n"
        "{prompt}\n\nSolution:"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MODELO PARA INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_eval(model_path: str, config: dict = None):
    """
    Carga modelo para inferencia (modo eval, sin gradientes).

    Args:
        model_path: ruta local al modelo entrenado o nombre HuggingFace
        config: configuración opcional

    Returns:
        (model, tokenizer)
    """
    cfg = {**EVAL_CONFIG, **(config or {})}
    print(f"\n[Eval] Cargando modelo: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )

    # Modo inferencia — más rápido, sin gradientes
    FastLanguageModel.for_inference(model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"  Modelo listo para inferencia.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR DATASET DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_dataset(domain: str, max_samples: int = 200):
    """
    Carga el split de test del dataset correspondiente al dominio.
    Siempre usamos el split de test para evaluación — nunca el de entrenamiento.
    """
    print(f"\n[Eval] Cargando dataset de evaluación: {domain}")

    if domain == "math":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        def format_fn(ex):
            return {
                "prompt": DOMAIN_PROMPTS["math"].format(question=ex["question"]),
                "answer": ex["answer"].split("####")[-1].strip(),
            }

    elif domain == "logic":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        def format_fn(ex):
            choices_text = "\n".join([
                f"  {label}. {text}"
                for label, text in zip(
                    ex["choices"]["label"],
                    ex["choices"]["text"]
                )
            ])
            return {
                "prompt": DOMAIN_PROMPTS["logic"].format(
                    question=ex["question"],
                    choices=choices_text
                ),
                "answer": ex["answerKey"],
            }

    elif domain == "code":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        def format_fn(ex):
            return {
                "prompt": DOMAIN_PROMPTS["code"].format(prompt=ex["prompt"]),
                "answer": ex["canonical_solution"],
                "test": ex.get("test", ""),
                "entry_point": ex.get("entry_point", ""),
            }

    else:
        raise ValueError(f"Dominio '{domain}' no válido.")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
    print(f"  Samples de evaluación: {len(dataset)}")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE RESPUESTAS
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """
    Genera una respuesta para un prompt dado.

    Args:
        do_sample: False = greedy (para Pass@1), True = sampling (para Pass@K)
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512 - max_new_tokens,
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Solo decodificar los tokens nuevos (no el prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_k_responses(
    model,
    tokenizer,
    prompt: str,
    k: int = 10,
    config: dict = None,
) -> List[str]:
    """
    Genera K respuestas para el mismo prompt con sampling.
    Usado para calcular Pass@K.
    """
    cfg = {**EVAL_CONFIG, **(config or {})}
    responses = []
    for _ in range(k):
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=True,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        responses.append(response)
    return responses


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICADORES DE CORRECTITUD POR DOMINIO
# ─────────────────────────────────────────────────────────────────────────────

def check_math_answer(response: str, gold_answer: str) -> bool:
    """
    Verifica si la respuesta matemática es correcta.
    Busca el número final en la respuesta y lo compara con la gold answer.
    """
    import re

    # Limpiar gold answer
    gold = gold_answer.strip().replace(",", "")

    # Extraer números de la respuesta (el último número suele ser la respuesta final)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response.replace(",", ""))
    if not numbers:
        return False

    # Comparar el último número encontrado con la gold answer
    last_number = numbers[-1]
    try:
        return abs(float(last_number) - float(gold)) < 1e-3
    except ValueError:
        return last_number == gold


def check_logic_answer(response: str, gold_answer: str) -> bool:
    """
    Verifica si la respuesta de opción múltiple es correcta.
    Busca la letra de la opción (A, B, C, D) en la respuesta.
    """
    import re
    gold = gold_answer.strip().upper()
    # Buscar la letra al inicio de la respuesta o después de "Answer:"
    matches = re.findall(r'\b([A-D])\b', response.upper())
    if matches:
        return matches[0] == gold
    return False


def check_code_answer(response: str, gold_answer: str, test: str = "", entry_point: str = "") -> bool:
    """
    Verifica si el código generado es correcto.
    Estrategia simple: verifica si contiene las palabras clave de la solución.
    Para evaluación más rigurosa se puede ejecutar los tests.
    """
    # Verificación simple por contenido
    gold_keywords = set(gold_answer.strip().split()[:10])
    response_keywords = set(response.strip().split()[:50])
    overlap = len(gold_keywords & response_keywords) / max(len(gold_keywords), 1)
    return overlap > 0.3


CHECKERS = {
    "math": check_math_answer,
    "logic": check_logic_answer,
    "code": check_code_answer,
}


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS: PASS@1 Y PASS@K
# ─────────────────────────────────────────────────────────────────────────────

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calcula Pass@K usando la fórmula unbiased de Chen et al. (HumanEval):
        Pass@K = 1 - C(n-c, k) / C(n, k)

    Args:
        n: número total de muestras generadas
        c: número de muestras correctas
        k: K para Pass@K

    Returns:
        Probabilidad de que al menos 1 de K muestras sea correcta.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    domain: str,
    max_samples: int = 200,
    k: int = 10,
    config: dict = None,
    save_results: bool = True,
    results_dir: str = "results",
) -> Dict:
    """
    Evaluación completa de un modelo en un dominio.

    Calcula:
        - Pass@1  (greedy decoding)
        - Pass@K  (sampling, K=10 por defecto)
        - Métricas de respuesta: longitud media, tasa de respuestas vacías

    Args:
        model_path: ruta al modelo o nombre HuggingFace
        domain: dominio de evaluación
        max_samples: máximo de ejemplos a evaluar
        k: K para Pass@K
        save_results: si True, guarda JSON con resultados

    Returns:
        Dict con todas las métricas.
    """
    cfg = {**EVAL_CONFIG, **(config or {})}

    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {model_path}")
    print(f"Dominio: {domain} | Pass@1 + Pass@{k}")
    print(f"{'='*60}")

    # Cargar modelo y dataset
    model, tokenizer = load_model_for_eval(model_path, cfg)
    dataset = load_eval_dataset(domain, max_samples)
    checker = CHECKERS[domain]

    results_pass1 = []
    results_passk = []
    response_lengths = []
    empty_responses = 0
    per_sample = []

    for i, sample in enumerate(dataset):
        prompt = sample["prompt"]
        gold = sample["answer"]
        extra = {k: v for k, v in sample.items() if k not in ["prompt", "answer"]}

        if (i + 1) % 20 == 0:
            p1_so_far = sum(results_pass1) / max(len(results_pass1), 1)
            print(f"  [{i+1}/{len(dataset)}] Pass@1 parcial: {p1_so_far:.3f}")

        # ── Pass@1: greedy ────────────────────────────────────
        response_greedy = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
        )

        if not response_greedy.strip():
            empty_responses += 1

        correct_p1 = checker(response_greedy, gold, **extra)
        results_pass1.append(int(correct_p1))
        response_lengths.append(len(response_greedy.split()))

        # ── Pass@K: sampling ──────────────────────────────────
        k_responses = generate_k_responses(model, tokenizer, prompt, k=k, config=cfg)
        n_correct = sum(checker(r, gold, **extra) for r in k_responses)
        pass_k = compute_pass_at_k(n=k, c=n_correct, k=k)
        results_passk.append(pass_k)

        # Guardar detalle por sample
        per_sample.append({
            "index": i,
            "prompt": prompt[:100] + "...",
            "gold": gold,
            "response_greedy": response_greedy[:200],
            "correct_p1": correct_p1,
            "pass_at_k": pass_k,
            "response_length": len(response_greedy.split()),
        })

    # ── Métricas finales ──────────────────────────────────────
    n = len(dataset)
    metrics = {
        "model_path": model_path,
        "domain": domain,
        "n_samples": n,
        "pass_at_1": sum(results_pass1) / n,
        "pass_at_k": sum(results_passk) / n,
        "k": k,
        "mean_response_length": sum(response_lengths) / n,
        "empty_response_rate": empty_responses / n,
    }

    print(f"\n{'─'*40}")
    print(f"RESULTADOS FINALES — {domain.upper()}")
    print(f"{'─'*40}")
    print(f"  Pass@1:              {metrics['pass_at_1']:.4f} ({metrics['pass_at_1']*100:.1f}%)")
    print(f"  Pass@{k}:             {metrics['pass_at_k']:.4f} ({metrics['pass_at_k']*100:.1f}%)")
    print(f"  Longitud media:      {metrics['mean_response_length']:.1f} tokens")
    print(f"  Respuestas vacías:   {metrics['empty_response_rate']*100:.1f}%")
    print(f"{'─'*40}")

    # ── Guardar resultados ────────────────────────────────────
    if save_results:
        Path(results_dir).mkdir(exist_ok=True)
        model_id = Path(model_path).name if Path(model_path).exists() else model_path.replace("/", "_")
        out_path = Path(results_dir) / f"{model_id}__{domain}.json"

        with open(out_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "per_sample": per_sample,
            }, f, indent=2)

        print(f"\n  Resultados guardados: {out_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results_dir: str = "results"):
    """
    Lee todos los JSON de resultados y construye una tabla comparativa
    en formato CSV y Markdown para el paper.
    """
    results_path = Path(results_dir)
    all_results = []

    for json_file in results_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        all_results.append(metrics)

    if not all_results:
        print("No hay resultados todavía. Corre evaluate.py primero.")
        return

    # Ordenar por dominio y modelo
    all_results.sort(key=lambda x: (x.get("domain", ""), x.get("model_path", "")))

    # Imprimir tabla Markdown
    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("="*80)
    print(f"{'Modelo':<45} {'Dominio':<8} {'Pass@1':>8} {'Pass@10':>8} {'Len':>6}")
    print("-"*80)

    for r in all_results:
        model_id = Path(r.get("model_path", "?")).name[:44]
        domain = r.get("domain", "?")
        p1 = r.get("pass_at_1", 0)
        pk = r.get("pass_at_k", 0)
        length = r.get("mean_response_length", 0)
        print(f"{model_id:<45} {domain:<8} {p1:>7.3f} {pk:>8.3f} {length:>6.1f}")

    print("="*80)

    # Guardar CSV
    import csv
    csv_path = results_path / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_path", "domain", "pass_at_1", "pass_at_k",
            "k", "mean_response_length", "empty_response_rate", "n_samples"
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nTabla CSV guardada: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label-Free RLVR Evaluation")

    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Ruta al modelo entrenado o nombre HuggingFace"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        choices=["math", "logic", "code", "all"],
        help="Dominio de evaluación. 'all' evalúa los 3 dominios (cross-domain)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Máximo de samples a evaluar (default: 200)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K para Pass@K (default: 10)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directorio donde guardar los resultados"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Solo construir tabla comparativa con resultados existentes"
    )

    args = parser.parse_args()

    if args.table:
        build_comparison_table(args.results_dir)
    elif args.domain == "all":
        # Evaluación cross-domain completa
        print(f"\nEvaluación cross-domain: {args.model_path}")
        all_metrics = {}
        for domain in ["math", "logic", "code"]:
            metrics = evaluate(
                model_path=args.model_path,
                domain=domain,
                max_samples=args.max_samples,
                k=args.k,
                results_dir=args.results_dir,
            )
            all_metrics[domain] = metrics

        print(f"\n{'='*50}")
        print("RESUMEN CROSS-DOMAIN")
        print(f"{'='*50}")
        for domain, m in all_metrics.items():
            print(f"  {domain:<8} Pass@1={m['pass_at_1']:.3f}  Pass@{args.k}={m['pass_at_k']:.3f}")
    else:
        evaluate(
            model_path=args.model_path,
            domain=args.domain,
            max_samples=args.max_samples,
            k=args.k,
            results_dir=args.results_dir,
        )
