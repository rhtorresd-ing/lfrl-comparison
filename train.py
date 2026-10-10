"""
train.py
========
Pipeline de entrenamiento GRPO con señales de recompensa intrínseca.

Uso:
    python train.py --signal self_certainty --domain math
    python train.py --signal entropy --domain logic
    python train.py --signal majority_voting --domain code
    python train.py --signal grpo_gold --domain math   # baseline con etiquetas

Autor: Rhonald Jose Torres Diaz
Afiliación: Universidad Nacional de Colombia — Sede La Paz
"""

import os
import csv
import json
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from rewards import self_certainty, entropy_reward, majority_voting
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN BASE
# ─────────────────────────────────────────────────────────────────────────────

BASE_CONFIG = {
    # Modelo
    "model_name": "Qwen/Qwen2.5-1.5B",
    "max_seq_length": 512,
    "load_in_4bit": True,          # QLoRA 4-bit — cabe en 8GB VRAM

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],

    # GRPO
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,   # batch efectivo = 8
    "learning_rate": 5e-6,
    "max_grad_norm": 0.1,
    "warmup_ratio": 0.1,
    "num_generations": 8,               # N muestras para majority voting y ventaja relativa
    "max_new_tokens": 256,

    # Logging
    "logging_steps": 10,
    "save_steps": 100,
    "output_dir": "runs",
}

# Datasets por dominio
DOMAIN_DATASETS = {
    "math":  ("openai/gsm8k", "main"),
    "logic": ("allenai/ai2_arc", "ARC-Challenge"),
    "code":  ("openai/openai_humaneval", None),
}

# Prompts por dominio
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
# BLOQUE 1: CARGAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: dict):
    """
    Carga Qwen2.5-1.5B con QLoRA 4-bit usando Unsloth.

    Unsloth aplica optimizaciones de memoria que permiten entrenar
    en 8GB VRAM. QLoRA congela los pesos originales y solo entrena
    matrices de bajo rango (LoRA adapters).
    """
    print(f"\n[Bloque 1] Cargando modelo: {config['model_name']}")
    print(f"  Cuantización 4-bit: {config['load_in_4bit']}")
    print(f"  Max seq length: {config['max_seq_length']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,  # Auto-detect: float16 en GPU
    )

    # Agregar adaptadores LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["lora_target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Reduce VRAM ~30%
        random_state=42,
    )

    # Configurar tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Reporte de parámetros
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 2: CARGAR DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_domain_dataset(domain: str, max_samples: int = 500):
    """
    Carga el dataset correspondiente al dominio y lo formatea
    con el prompt template adecuado.

    Limitamos a max_samples para que el entrenamiento sea manejable
    en una GPU de 8GB con tiempo razonable (~2-4 horas por run).
    """
    print(f"\n[Bloque 2] Cargando dataset para dominio: {domain}")

    dataset_name, subset = DOMAIN_DATASETS[domain]
    prompt_template = DOMAIN_PROMPTS[domain]

    if subset:
        dataset = load_dataset(dataset_name, subset, split="train")
    else:
        dataset = load_dataset(dataset_name, split="test")

    # Limitar samples
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    print(f"  Dataset: {dataset_name} ({subset or 'default'})")
    print(f"  Samples: {len(dataset)}")

    # Formatear según dominio
    def format_math(example):
        return {
            "prompt": prompt_template.format(question=example["question"]),
            "answer": example["answer"],
        }

    def format_logic(example):
        choices_text = "\n".join([
            f"  {label}. {text}"
            for label, text in zip(
                example["choices"]["label"],
                example["choices"]["text"]
            )
        ])
        return {
            "prompt": prompt_template.format(
                question=example["question"],
                choices=choices_text
            ),
            "answer": example["answerKey"],
        }

    def format_code(example):
        return {
            "prompt": prompt_template.format(prompt=example["prompt"]),
            "answer": example["canonical_solution"],
        }

    formatters = {
        "math": format_math,
        "logic": format_logic,
        "code": format_code,
    }

    dataset = dataset.map(formatters[domain], remove_columns=dataset.column_names)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 3: REWARD FUNCTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def build_reward_function(signal: str, tokenizer, model_ref=None):
    """
    Construye la función de reward compatible con GRPOTrainer.

    GRPOTrainer espera una función con firma:
        reward_fn(prompts, completions, **kwargs) -> List[float]

    Esta función adapta nuestras señales intrínsecas a esa interfaz.

    Args:
        signal: "self_certainty" | "entropy" | "majority_voting" | "grpo_gold"
        tokenizer: tokenizer del modelo
        model_ref: referencia al modelo (necesaria para obtener logits)
    """
    print(f"\n[Bloque 3] Señal de recompensa: {signal}")

    def reward_self_certainty(prompts, completions, **kwargs):
        """Reward basado en self-certainty de los logits."""
        rewards = []
        for completion in completions:
            # Tokenizar completion
            inputs = tokenizer(
                completion, return_tensors="pt",
                truncation=True, max_length=256
            ).to("cuda")

            with torch.no_grad():
                outputs = model_ref(**inputs)
                logits = outputs.logits[0]  # (seq_len, vocab_size)

            reward = self_certainty(logits).item()
            rewards.append(reward)
        return rewards

    def reward_entropy(prompts, completions, **kwargs):
        """Reward basado en entropía negativa."""
        rewards = []
        for completion in completions:
            inputs = tokenizer(
                completion, return_tensors="pt",
                truncation=True, max_length=256
            ).to("cuda")

            with torch.no_grad():
                outputs = model_ref(**inputs)
                logits = outputs.logits[0]

            reward = entropy_reward(logits).item()
            rewards.append(reward)
        return rewards

    def reward_majority_voting(prompts, completions, **kwargs):
        """
        Reward por mayoría de votos.
        completions ya son las N generaciones del mismo prompt
        agrupadas por GRPOTrainer.
        """
        rewards = majority_voting(completions).tolist()
        return rewards

    def reward_grpo_gold(prompts, completions, **kwargs):
        """
        Baseline: GRPO con gold labels.
        Verifica si la completion contiene la respuesta correcta.
        Requiere que kwargs contenga 'answers'.
        """
        answers = kwargs.get("answers", [""] * len(completions))
        rewards = []
        for completion, answer in zip(completions, answers):
            # Reward binario: 1 si contiene la respuesta, 0 si no
            reward = 1.0 if str(answer).lower() in completion.lower() else 0.0
            rewards.append(reward)
        return rewards

    signal_map = {
        "self_certainty": reward_self_certainty,
        "entropy": reward_entropy,
        "majority_voting": reward_majority_voting,
        "grpo_gold": reward_grpo_gold,
    }

    if signal not in signal_map:
        raise ValueError(f"Señal '{signal}' no válida. Opciones: {list(signal_map.keys())}")

    return signal_map[signal]


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 + 5: GRPO TRAINING LOOP + LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    signal: str,
    domain: str,
    max_samples: int = 500,
    use_wandb: bool = False,
    config: dict = None,
):
    """
    Pipeline completo de entrenamiento GRPO label-free.

    Args:
        signal: señal de recompensa intrínseca a usar
        domain: dominio de entrenamiento (math/logic/code)
        max_samples: máximo de ejemplos de entrenamiento
        use_wandb: si True, loguea métricas a wandb.ai
        config: configuración personalizada (opcional)
    """
    cfg = {**BASE_CONFIG, **(config or {})}

    # Nombre del run
    run_name = f"{signal}__{domain}__{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(cfg["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RUN: {run_name}")
    print(f"{'='*60}")

    # Guardar config
    with open(output_dir / "config.json", "w") as f:
        json.dump({**cfg, "signal": signal, "domain": domain}, f, indent=2)

    # ── Wandb ──────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project="lfrl-comparison",
            name=run_name,
            config={**cfg, "signal": signal, "domain": domain}
        )

    # ── Bloque 1: Modelo ───────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Bloque 2: Dataset ──────────────────────────────────────
    dataset = load_domain_dataset(domain, max_samples)

    # ── Bloque 3: Reward function ──────────────────────────────
    reward_fn = build_reward_function(signal, tokenizer, model)

    # ── Bloque 4: GRPO Config ──────────────────────────────────
    print(f"\n[Bloque 4] Configurando GRPOTrainer")

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_grad_norm=cfg["max_grad_norm"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        num_generations=cfg["num_generations"],

        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        bf16=True,
        dataloader_num_workers=0,
    )

    # ── Bloque 4: Trainer ──────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # ── CSV Logger personalizado ───────────────────────────────
    csv_path = output_dir / "training_log.csv"
    csv_fields = [
        "step", "loss", "reward_mean", "reward_std",
        "kl_penalty", "response_length", "signal", "domain"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    original_log = trainer.log

    def custom_log(logs, *args, **kwargs):
        """Intercepta logs de Trainer y guarda en CSV."""
        original_log(logs, *args, **kwargs)

        step = trainer.state.global_step
        row = {
            "step": step,
            "loss": logs.get("loss", ""),
            "reward_mean": logs.get("reward_mean", logs.get("rewards/mean", "")),
            "reward_std": logs.get("reward_std", logs.get("rewards/std", "")),
            "kl_penalty": logs.get("kl_penalty", logs.get("kl", "")),
            "response_length": logs.get("response_length", logs.get("completions/mean_length", "")),
            "signal": signal,
            "domain": domain,
        }
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

    trainer.log = custom_log

    # ── Bloque 5: Entrenar ─────────────────────────────────────
    print(f"\n[Bloque 5] Iniciando entrenamiento...")
    print(f"  Steps estimados: {len(dataset) // (cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'])}")
    print(f"  Output dir: {output_dir}")
    print(f"  CSV log: {csv_path}\n")

    trainer.train()

    # ── Guardar modelo final ───────────────────────────────────
    print(f"\n[Final] Guardando modelo en {output_dir}/final_model")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    if use_wandb:
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO COMPLETADO: {run_name}")
    print(f"Modelo guardado en: {output_dir}/final_model")
    print(f"Log CSV: {csv_path}")
    print(f"{'='*60}")

    return str(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label-Free RLVR Training")

    parser.add_argument(
        "--signal",
        type=str,
        default="self_certainty",
        choices=["self_certainty", "entropy", "majority_voting", "grpo_gold"],
        help="Señal de recompensa intrínseca a usar"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        choices=["math", "logic", "code"],
        help="Dominio de entrenamiento"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Máximo de ejemplos de entrenamiento (default: 500)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Activar logging a wandb.ai"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Solo verifica imports y config sin entrenar"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — verificando configuración...")
        model, tokenizer = load_model_and_tokenizer(BASE_CONFIG)
        dataset = load_domain_dataset(args.domain, max_samples=5)
        reward_fn = build_reward_function(args.signal, tokenizer, model)
        print(f"\nConfig OK:")
        print(f"  Señal:   {args.signal}")
        print(f"  Dominio: {args.domain}")
        print(f"  Dataset: {len(dataset)} samples (dry run: 5)")
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print("\nListo para entrenar. Corre sin --dry_run para iniciar.")
    else:
        train(
            signal=args.signal,
            domain=args.domain,
            max_samples=args.max_samples,
            use_wandb=args.wandb,
        )
