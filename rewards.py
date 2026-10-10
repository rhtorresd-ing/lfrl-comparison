"""
rewards.py
==========
Tres funciones de recompensa intrínseca para Label-Free RLVR.

Señales implementadas:
  1. self_certainty   — KL divergence del output respecto a distribución uniforme
  2. entropy_reward   — entropía negativa de la secuencia generada
  3. majority_voting  — consenso entre N muestras del mismo prompt

Autor: Rhonald Jose Torres Diaz
Afiliación: Universidad Nacional de Colombia — Sede La Paz
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES COMPARTIDAS
# ─────────────────────────────────────────────────────────────────────────────

def _get_token_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convierte logits a distribución de probabilidades por token.

    Args:
        logits: Tensor de forma (seq_len, vocab_size) o (batch, seq_len, vocab_size)

    Returns:
        Tensor de probabilidades con la misma forma que logits.
    """
    return F.softmax(logits, dim=-1)


def _kl_from_uniform(probs: torch.Tensor) -> torch.Tensor:
    """
    Calcula KL(p || uniform) para una distribución de probabilidades.

    KL(p || u) = sum_i p_i * log(p_i * vocab_size)
               = sum_i p_i * log(p_i) + log(vocab_size)
               = -H(p) + log(vocab_size)

    Un valor alto significa que el modelo está muy seguro (concentrado).
    Un valor de 0 significa distribución completamente uniforme (máxima incertidumbre).

    Args:
        probs: Tensor de probabilidades (..., vocab_size)

    Returns:
        Escalar KL divergence promediado sobre todos los tokens.
    """
    vocab_size = probs.shape[-1]
    # Clamp para evitar log(0)
    probs_clamped = probs.clamp(min=1e-10)
    # KL(p || uniform) = sum p * log(p * V)
    kl = (probs_clamped * (torch.log(probs_clamped) + torch.log(torch.tensor(vocab_size, dtype=torch.float)))).sum(dim=-1)
    return kl.mean()


# ─────────────────────────────────────────────────────────────────────────────
# SEÑAL 1: SELF-CERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

def self_certainty(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calcula la self-certainty como KL divergence promedio entre la distribución
    de output del modelo y una distribución uniforme sobre el vocabulario.

    Basado en INTUITOR (Zhao et al., ICLR 2026):
        r = mean_t KL(p_t || uniform)

    Un reward más alto = el modelo está más seguro de sus respuestas.
    Esto incentiva al modelo a generar outputs más deterministas y confiados.

    Args:
        logits: Tensor (seq_len, vocab_size) o (batch_size, seq_len, vocab_size)
                Logits crudos del modelo para cada token generado.
        attention_mask: Tensor opcional (batch_size, seq_len) para ignorar
                        tokens de padding en el cálculo.

    Returns:
        Tensor escalar con el reward de self-certainty.

    Ejemplo:
        >>> logits = torch.randn(10, 32000)  # 10 tokens, vocab 32k
        >>> reward = self_certainty(logits)
        >>> print(reward)  # valor positivo, mayor = más seguro
    """
    probs = _get_token_probs(logits)

    if attention_mask is not None and logits.dim() == 3:
        # Aplicar mask: ignorar tokens de padding
        mask = attention_mask.unsqueeze(-1).float()
        probs = probs * mask
        probs_clamped = probs.clamp(min=1e-10)
        vocab_size = probs.shape[-1]
        kl_per_token = (probs_clamped * (torch.log(probs_clamped) + torch.log(
            torch.tensor(vocab_size, dtype=torch.float, device=logits.device)
        ))).sum(dim=-1)
        # Promedio solo sobre tokens no enmascarados
        n_tokens = attention_mask.float().sum()
        reward = (kl_per_token * attention_mask.float()).sum() / n_tokens.clamp(min=1)
    else:
        reward = _kl_from_uniform(probs)

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# SEÑAL 2: ENTROPY MINIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def entropy_reward(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Calcula la recompensa basada en entropía negativa de la secuencia generada.

    r = -mean_t H(p_t)
    donde H(p_t) = -sum_i p_i * log(p_i)

    Recompensar entropía negativa equivale a penalizar la incertidumbre:
    el modelo aprende a generar tokens con distribuciones más concentradas.

    Relación con self_certainty:
        self_certainty = entropy_reward + log(vocab_size)
    Son equivalentes hasta una constante, pero entropy_reward es más
    interpretable como métrica (está en nats).

    Args:
        logits: Tensor (seq_len, vocab_size) o (batch_size, seq_len, vocab_size)
        attention_mask: Mask opcional para ignorar padding.
        normalize: Si True, normaliza la entropía por log(vocab_size) para
                   obtener valores en [-1, 0]. Útil para comparar entre
                   modelos con distintos tamaños de vocabulario.

    Returns:
        Tensor escalar. Valores más altos (menos negativos) = menor entropía
        = mayor certeza del modelo.

    Ejemplo:
        >>> logits = torch.randn(10, 32000)
        >>> reward = entropy_reward(logits)
        >>> print(reward)  # negativo, más cercano a 0 = mejor
    """
    probs = _get_token_probs(logits)
    probs_clamped = probs.clamp(min=1e-10)

    # H(p) = -sum p * log(p)
    entropy_per_token = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)

    if normalize:
        vocab_size = logits.shape[-1]
        entropy_per_token = entropy_per_token / torch.log(torch.tensor(vocab_size, dtype=torch.float))

    if attention_mask is not None and logits.dim() == 3:
        n_tokens = attention_mask.float().sum()
        mean_entropy = (entropy_per_token * attention_mask.float()).sum() / n_tokens.clamp(min=1)
    else:
        mean_entropy = entropy_per_token.mean()

    # Negamos para que mayor reward = menor entropía
    return -mean_entropy


# ─────────────────────────────────────────────────────────────────────────────
# SEÑAL 3: MAJORITY VOTING
# ─────────────────────────────────────────────────────────────────────────────

def majority_voting(
    responses: List[str],
    normalize: bool = True
) -> torch.Tensor:
    """
    Calcula la recompensa por votación de mayoría sobre N respuestas generadas
    para el mismo prompt.

    Algoritmo:
        1. Generar N respuestas para el mismo prompt (sin etiquetas)
        2. Normalizar cada respuesta (strip, lowercase)
        3. Contar frecuencias
        4. Asignar reward = freq(respuesta) / N a cada respuesta
           Las respuestas en consenso reciben reward alto,
           las respuestas atípicas reciben reward bajo.

    Basado en TTRL (Zuo et al., 2025): usa el consenso de muestras
    como pseudo-etiqueta sin ninguna supervisión externa.

    Args:
        responses: Lista de N strings con las respuestas generadas para
                   un mismo prompt. Típicamente N=8 en los experimentos.
        normalize: Si True, el reward es freq/N (entre 0 y 1).
                   Si False, el reward es la frecuencia absoluta.

    Returns:
        Tensor de forma (N,) con el reward de cada respuesta.
        La respuesta más común tendrá el reward más alto.

    Ejemplo:
        >>> responses = ["4", "4", "4", "5", "4", "3", "4", "5"]
        >>> rewards = majority_voting(responses)
        >>> print(rewards)
        # tensor([0.625, 0.625, 0.625, 0.25, 0.625, 0.125, 0.625, 0.25])
    """
    n = len(responses)
    if n == 0:
        return torch.tensor([])

    # Normalizar respuestas para comparación
    normalized = [r.strip().lower() for r in responses]

    # Contar frecuencias
    freq = Counter(normalized)

    # Asignar reward a cada respuesta
    rewards = []
    for resp in normalized:
        count = freq[resp]
        reward = count / n if normalize else float(count)
        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN WRAPPER: seleccionar señal por nombre
# ─────────────────────────────────────────────────────────────────────────────

def get_reward_fn(signal_name: str):
    """
    Retorna la función de recompensa correspondiente al nombre dado.

    Args:
        signal_name: "self_certainty" | "entropy" | "majority_voting"

    Returns:
        Función de recompensa callable.

    Raises:
        ValueError si el nombre no es válido.
    """
    registry = {
        "self_certainty": self_certainty,
        "entropy": entropy_reward,
        "majority_voting": majority_voting,
    }
    if signal_name not in registry:
        raise ValueError(
            f"Señal '{signal_name}' no reconocida. "
            f"Opciones válidas: {list(registry.keys())}"
        )
    return registry[signal_name]


# ─────────────────────────────────────────────────────────────────────────────
# TESTS UNITARIOS (ejecutar con: python rewards.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: rewards.py — señales de recompensa intrínseca")
    print("=" * 60)

    torch.manual_seed(42)

    # ── Test 1: self_certainty ──────────────────────────────────
    print("\n[1] self_certainty")
    # Caso A: logits muy concentrados (modelo seguro)
    logits_sure = torch.zeros(5, 100)
    logits_sure[:, 0] = 10.0  # token 0 con probabilidad ~1
    r_sure = self_certainty(logits_sure)

    # Caso B: logits uniformes (modelo inseguro)
    logits_uniform = torch.zeros(5, 100)
    r_uniform = self_certainty(logits_uniform)

    print(f"  Modelo seguro    → reward = {r_sure.item():.4f}")
    print(f"  Modelo inseguro  → reward = {r_uniform.item():.4f}")
    assert r_sure > r_uniform, "FAIL: modelo seguro deberia tener reward mayor"
    print("  ✓ PASS: modelo seguro > modelo inseguro")

    # ── Test 2: entropy_reward ──────────────────────────────────
    print("\n[2] entropy_reward")
    r_ent_sure    = entropy_reward(logits_sure)
    r_ent_uniform = entropy_reward(logits_uniform)
    print(f"  Modelo seguro    → reward = {r_ent_sure.item():.4f}")
    print(f"  Modelo inseguro  → reward = {r_ent_uniform.item():.4f}")
    assert r_ent_sure > r_ent_uniform, "FAIL: entropía negativa debería ser mayor cuando más seguro"
    print("  ✓ PASS: entropía negativa mayor en modelo seguro")

    # ── Test 3: majority_voting ─────────────────────────────────
    print("\n[3] majority_voting")
    responses = ["4", "4", "4", "5", "4", "3", "4", "5"]
    rewards = majority_voting(responses)
    print(f"  Respuestas: {responses}")
    print(f"  Rewards:    {rewards.tolist()}")
    assert rewards[0].item() == rewards[1].item(), "FAIL: respuestas iguales deben tener mismo reward"
    assert rewards[0] > rewards[3], "FAIL: '4' debe tener mayor reward que '5'"
    print("  ✓ PASS: consenso asignado correctamente")

    # ── Test 4: get_reward_fn ───────────────────────────────────
    print("\n[4] get_reward_fn")
    for name in ["self_certainty", "entropy", "majority_voting"]:
        fn = get_reward_fn(name)
        print(f"  '{name}' → {fn.__name__} ✓")

    try:
        get_reward_fn("invalid")
    except ValueError as e:
        print(f"  Error esperado capturado: {e} ✓")

    print("\n" + "=" * 60)
    print("TODOS LOS TESTS PASARON ✓")
    print("=" * 60)