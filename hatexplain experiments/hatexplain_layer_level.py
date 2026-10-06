"""HateXplain experiment with layer-level lambda-balanced attention initialization.

This version uses one QK circuit and one OV circuit per attention layer, implemented
with PyTorch nn.Linear modules. There is no per-head balancing. The same code can
run either a one-layer or a two-layer attention classifier.

Install:
    python -m pip install torch numpy "transformers>=4.46,<5"

Example: one layer
    python hatexplain_layer_level_updated.py \
        --n-layers 1 --epochs 200 \
        --seeds 1234 1235 1236 1237 1238 \
        --lambda-qk 200 --lambda-ov 40 \
        --output-dir runs/one_layer_qk200_ov40

Example: two layers
    python hatexplain_layer_level_updated.py \
        --n-layers 2 --epochs 200 \
        --seeds 1234 1235 1236 1237 1238 \
        --lambda-qk 200 --lambda-ov 40 \
        --output-dir runs/two_layer_qk200_ov40

Balanced comparison:
    Keep every option, seed, and sigma fixed and change only --lambda-qk and
    --lambda-ov. For a fixed seed, each layer uses the same initial effective QK
    and OV products for every lambda pair.

Architecture:
  * Each attention layer contains
        key_layer:    R^{d_model} -> R^{d_head}
        query_layer:  R^{d_model} -> R^{d_head}
        value_layer:  R^{d_model} -> R^{d_value}
        output_layer: R^{d_value} -> R^{d_model}
  * The HateXplain classifier is a separate final map R^{d_model} -> R^3.
  * Each layer has a residual connection. There is no MLP or LayerNorm, matching
    the intentionally simple attention-only setup used for the real-data test.
  * In the two-layer model, explanation scores average attention across layers.

Initialization conventions:
  * QK balance:
        W_K^T W_K - W_Q^T W_Q = lambda_qk I_{d_head}
    while preserving the effective product
        C_QK = W_K W_Q^T.

  * OV balance:
        W_O^T W_O - W_V W_V^T = lambda_ov I_{d_value}
    while preserving the effective product
        C_OV = W_O W_V.

  * Positive lambda_qk is key-heavy; positive lambda_ov is output-heavy.
  * The SVD constructions are performed in float64 and copied into the existing
    nn.Linear parameters.
  * Initialization audits report both balance error and effective-product error.
  * Training drift is measured relative to the actual initial Gram differences.

Evaluation conventions are inherited from the supplied HateXplain experiment,
with rationale mass computed from all valid rationale annotations:
  * Labels: hatespeech=0, normal=1, offensive=2; tied votes are skipped by
    default (--label-tie-policy can select lowest or error).
  * For an example with A valid rationale annotations r^(1),...,r^(A), rationale
    mass is computed for every annotation and averaged across annotations. By
    linearity, this is implemented using the soft mean rationale weight
        r_bar = (1/A) * sum_a r^(a)
    rather than thresholding the annotations into a consensus mask.
  * Original words are aligned to WordPieces with is_split_into_words=True.
  * Attention is averaged over layers and valid query positions, then normalized
    over valid keys INCLUDING special tokens.
  * rationale_mass averages the per-example annotation-averaged mass over
    examples with at least one valid rationale annotation.
  * rationale_mass_all_examples averages the same per-example quantity over the
    full test set; examples without valid rationale annotations contribute zero.
  * Faithfulness selects the top floor(fraction * content_length) WordPieces,
    at least one. PAD and inserted special tokens are not candidates.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Tuple
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def _balanced_singular_values(
    singular_values: np.ndarray,
    lmda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a,b satisfying a_i b_i=s_i and a_i^2-b_i^2=lambda.

    The computation avoids catastrophic cancellation when |lambda| is large.
    Here ``a`` is assigned to W_K or W_O, while ``b`` is assigned to W_Q or W_V.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    root = np.hypot(float(lmda), 2.0 * s)

    if lmda >= 0:
        a = np.sqrt(0.5 * (root + lmda))
        b = np.divide(s, a, out=np.zeros_like(s), where=a > 0)
    else:
        b = np.sqrt(0.5 * (root - lmda))
        a = np.divide(s, b, out=np.zeros_like(s), where=b > 0)

    return a, b


def get_lambda_balanced_query_key(
    lmda: float,
    d_model: int,
    d_head: int,
    sigma: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct PyTorch-compatible Q/K weights with fixed effective QK product.

    The theoretical matrices have shape
        W_K, W_Q: (d_model, d_head),
    while ``nn.Linear`` stores their transposes:
        key_weight, query_weight: (d_head, d_model).

    The construction preserves
        C_QK = W_K W_Q^T
    and imposes
        W_K^T W_K - W_Q^T W_Q = lambda I_{d_head}.
    """
    if d_head > d_model:
        raise ValueError("The QK construction requires d_head <= d_model.")

    rng = np.random.default_rng(seed)
    std = sigma * np.sqrt(2.0 / d_model)

    W_K_base = rng.standard_normal((d_model, d_head)).astype(np.float64) * std
    W_Q_base = rng.standard_normal((d_model, d_head)).astype(np.float64) * std

    C0 = W_K_base @ W_Q_base.T
    U, S, Vt = np.linalg.svd(C0, full_matrices=False)

    R_raw = rng.standard_normal((d_head, d_head)).astype(np.float64)
    R, _ = np.linalg.qr(R_raw)

    a, b = _balanced_singular_values(S[:d_head], lmda)

    W_K = U[:, :d_head] @ np.diag(a) @ R.T
    W_Q = Vt.T[:, :d_head] @ np.diag(b) @ R.T

    key_weight = W_K.T
    query_weight = W_Q.T

    return key_weight, query_weight, C0


def get_lambda_balanced_output_value(
    lmda: float,
    d_model: int,
    d_value: int,
    d_out: int,
    sigma: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct PyTorch-compatible O/V weights with fixed effective OV product.

    Shapes:
        W_V: (d_value, d_model)
        W_O: (d_out, d_value)
        C_OV = W_O W_V: (d_out, d_model)

    The construction preserves C_OV and imposes
        W_O^T W_O - W_V W_V^T = lambda I_{d_value}.

    In the attention layer below, d_out=d_model. The final three-class classifier
    is not part of the balanced OV factorization.
    """
    if d_value > min(d_model, d_out):
        raise ValueError(
            "The OV construction requires d_value <= min(d_model, d_out)."
        )

    rng = np.random.default_rng(seed)

    std_v = sigma * np.sqrt(2.0 / d_model)
    std_o = sigma * np.sqrt(2.0 / d_value)

    wv_base = (
        rng.standard_normal((d_value, d_model)).astype(np.float64) * std_v
    )
    wo_base = (
        rng.standard_normal((d_out, d_value)).astype(np.float64) * std_o
    )

    C0 = wo_base @ wv_base
    U, S, Vt = np.linalg.svd(C0, full_matrices=False)

    R_raw = rng.standard_normal((d_value, d_value)).astype(np.float64)
    R, _ = np.linalg.qr(R_raw)

    a, b = _balanced_singular_values(S[:d_value], lmda)

    init_wo = U[:, :d_value] @ np.diag(a) @ R.T
    init_wv = R @ np.diag(b) @ Vt[:d_value, :]

    return init_wo, init_wv, C0


class BalancedAttentionLayer(nn.Module):
    """One layer-level QK circuit and one layer-level OV circuit."""

    def __init__(
        self,
        d_model,
        d_head,
        d_value,
        sigma_qk,
        sigma_ov,
        seed,
        lambda_qk=0.01,
        lambda_ov=0.01,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()

        for name, value in (
            ("d_model", d_model),
            ("d_head", d_head),
            ("d_value", d_value),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if d_head > d_model:
            raise ValueError("d_head must be <= d_model.")
        if d_value > d_model:
            raise ValueError("d_value must be <= d_model.")
        if not all(
            math.isfinite(float(v)) and float(v) >= 0
            for v in (sigma_qk, sigma_ov)
        ):
            raise ValueError("sigma_qk and sigma_ov must be finite and nonnegative.")
        if not all(math.isfinite(float(v)) for v in (lambda_qk, lambda_ov)):
            raise ValueError("The imbalance values must be finite.")

        self.d_model = int(d_model)
        self.d_head = int(d_head)
        self.d_value = int(d_value)
        self.lambda_qk = float(lambda_qk)
        self.lambda_ov = float(lambda_ov)
        self.sigma_qk = float(sigma_qk)
        self.sigma_ov = float(sigma_ov)
        self.seed = int(seed)

        options = {"device": device, "dtype": dtype}

        self.key_layer = nn.Linear(d_model, d_head, bias=False, **options)
        self.query_layer = nn.Linear(d_model, d_head, bias=False, **options)
        self.value_layer = nn.Linear(d_model, d_value, bias=False, **options)
        self.output_layer = nn.Linear(d_value, d_model, bias=False, **options)

        ref_options = {"device": device, "dtype": torch.float64}
        self.register_buffer(
            "_target_qk_product",
            torch.zeros((d_model, d_model), **ref_options),
        )
        self.register_buffer(
            "_target_ov_product",
            torch.zeros((d_model, d_model), **ref_options),
        )
        self.register_buffer(
            "_initial_qk_balance",
            torch.zeros((d_head, d_head), **ref_options),
        )
        self.register_buffer(
            "_initial_ov_balance",
            torch.zeros((d_value, d_value), **ref_options),
        )
        self.register_buffer(
            "_initial_qk_gram_scale",
            torch.zeros((), **ref_options),
        )
        self.register_buffer(
            "_initial_ov_gram_scale",
            torch.zeros((), **ref_options),
        )

        self.initialize_balanced()

    @torch.no_grad()
    def initialize_balanced(self):
        """Initialize all four nn.Linear weights with fixed effective products."""
        init_wk, init_wq, target_qk = get_lambda_balanced_query_key(
            lmda=self.lambda_qk,
            d_model=self.d_model,
            d_head=self.d_head,
            sigma=self.sigma_qk,
            seed=self.seed,
        )
        init_wo, init_wv, target_ov = get_lambda_balanced_output_value(
            lmda=self.lambda_ov,
            d_model=self.d_model,
            d_value=self.d_value,
            d_out=self.d_model,
            sigma=self.sigma_ov,
            seed=self.seed,
        )

        assert init_wk.shape == (self.d_head, self.d_model)
        assert init_wq.shape == (self.d_head, self.d_model)
        assert init_wv.shape == (self.d_value, self.d_model)
        assert init_wo.shape == (self.d_model, self.d_value)

        self.key_layer.weight.copy_(
            torch.as_tensor(
                init_wk,
                dtype=self.key_layer.weight.dtype,
                device=self.key_layer.weight.device,
            )
        )
        self.query_layer.weight.copy_(
            torch.as_tensor(
                init_wq,
                dtype=self.query_layer.weight.dtype,
                device=self.query_layer.weight.device,
            )
        )
        self.value_layer.weight.copy_(
            torch.as_tensor(
                init_wv,
                dtype=self.value_layer.weight.dtype,
                device=self.value_layer.weight.device,
            )
        )
        self.output_layer.weight.copy_(
            torch.as_tensor(
                init_wo,
                dtype=self.output_layer.weight.dtype,
                device=self.output_layer.weight.device,
            )
        )

        self._target_qk_product.copy_(
            torch.as_tensor(target_qk, dtype=torch.float64, device=self._target_qk_product.device)
        )
        self._target_ov_product.copy_(
            torch.as_tensor(target_ov, dtype=torch.float64, device=self._target_ov_product.device)
        )

        self.capture_initial_balance()

    @torch.no_grad()
    def _grams(self):
        K = self.key_layer.weight.detach().to(dtype=torch.float64)
        Q = self.query_layer.weight.detach().to(dtype=torch.float64)
        V = self.value_layer.weight.detach().to(dtype=torch.float64)
        O = self.output_layer.weight.detach().to(dtype=torch.float64)

        # nn.Linear storage orientation:
        #   K,Q: (d_head,d_model), V: (d_value,d_model), O: (d_model,d_value)
        KK = K @ K.T
        QQ = Q @ Q.T
        OO = O.T @ O
        VV = V @ V.T
        return KK, QQ, OO, VV

    @torch.no_grad()
    def _effective_products(self):
        K = self.key_layer.weight.detach().to(dtype=torch.float64)
        Q = self.query_layer.weight.detach().to(dtype=torch.float64)
        V = self.value_layer.weight.detach().to(dtype=torch.float64)
        O = self.output_layer.weight.detach().to(dtype=torch.float64)

        C_qk = K.T @ Q
        C_ov = O @ V
        return C_qk, C_ov

    @torch.no_grad()
    def capture_initial_balance(self):
        KK, QQ, OO, VV = self._grams()
        self._initial_qk_balance.copy_(KK - QQ)
        self._initial_ov_balance.copy_(OO - VV)
        self._initial_qk_gram_scale.copy_(
            torch.linalg.matrix_norm(KK) + torch.linalg.matrix_norm(QQ)
        )
        self._initial_ov_gram_scale.copy_(
            torch.linalg.matrix_norm(OO) + torch.linalg.matrix_norm(VV)
        )

    @staticmethod
    def _safe_relative(drift, scale):
        if scale.item() > 0:
            return (drift / scale.clamp_min(torch.finfo(scale.dtype).tiny)).item()
        return None

    @torch.no_grad()
    def initialization_audit(self):
        """Audit requested balance and fixed-product preservation before training."""
        report = self.check_invariance_drift(detailed=True)
        C_qk, C_ov = self._effective_products()
        report["qk"]["effective_product_error"] = torch.linalg.matrix_norm(
            C_qk - self._target_qk_product
        ).item()
        report["ov"]["effective_product_error"] = torch.linalg.matrix_norm(
            C_ov - self._target_ov_product
        ).item()
        report["qk"]["effective_product_norm"] = torch.linalg.matrix_norm(
            self._target_qk_product
        ).item()
        report["ov"]["effective_product_norm"] = torch.linalg.matrix_norm(
            self._target_ov_product
        ).item()
        return report

    @torch.no_grad()
    def check_invariance_drift(self, detailed=False):
        KK, QQ, OO, VV = self._grams()
        qk = KK - QQ
        ov = OO - VV

        qk_drift = torch.linalg.matrix_norm(qk - self._initial_qk_balance)
        ov_drift = torch.linalg.matrix_norm(ov - self._initial_ov_balance)

        if not detailed:
            return qk_drift.item(), ov_drift.item()

        eye_qk = torch.eye(self.d_head, dtype=qk.dtype, device=qk.device)
        eye_ov = torch.eye(self.d_value, dtype=ov.dtype, device=ov.device)

        def report(current, initial, scale, drift, lmda, eye):
            return {
                "initial_target_error": torch.linalg.matrix_norm(
                    initial - lmda * eye
                ).item(),
                "drift": drift.item(),
                "relative_drift": self._safe_relative(drift, scale),
                "current_target_error": torch.linalg.matrix_norm(
                    current - lmda * eye
                ).item(),
            }

        return {
            "qk": report(
                qk,
                self._initial_qk_balance,
                self._initial_qk_gram_scale,
                qk_drift,
                self.lambda_qk,
                eye_qk,
            ),
            "ov": report(
                ov,
                self._initial_ov_balance,
                self._initial_ov_gram_scale,
                ov_drift,
                self.lambda_ov,
                eye_ov,
            ),
        }

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError("Input feature dimension does not match d_model.")

        if attention_mask is None:
            valid = torch.ones((B, L), dtype=torch.bool, device=x.device)
        else:
            if attention_mask.shape != (B, L):
                raise ValueError("attention_mask must have shape (batch, length).")
            valid = attention_mask.to(device=x.device, dtype=torch.bool)

        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        causal = torch.ones((L, L), dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, :, :] & valid[:, None, :]
        allowed = allowed & valid[:, :, None]

        has_key = allowed.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~allowed, float("-inf"))
        safe_scores = torch.where(
            has_key, masked_scores, torch.zeros_like(masked_scores)
        )
        attn_weights = F.softmax(safe_scores, dim=-1).masked_fill(~allowed, 0.0)

        context = attn_weights @ v
        out = self.output_layer(context)
        return out, attn_weights


class BalancedTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        d_head,
        d_value,
        n_layers=1,
        n_classes=3,
        max_seq_len=64,
        lambda_qk=0.01,
        lambda_ov=0.01,
        *,
        sigma_qk=0.01,
        sigma_ov=0.01,
        seed=42,
        pad_token_id=0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if n_layers not in (1, 2):
            raise ValueError("This experiment supports n_layers in {1, 2}.")

        self.pad_token_id = int(pad_token_id)
        self.max_seq_len = int(max_seq_len)
        self.n_layers = int(n_layers)

        options = {"device": device, "dtype": dtype}
        self.token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=self.pad_token_id, **options
        )
        self.pos_emb = nn.Embedding(max_seq_len, d_model, **options)

        # Layer 1 uses the experiment seed directly. Layer 2 uses a deterministic
        # offset so that its fixed effective products differ while remaining
        # identical across lambda comparisons for the same experiment seed.
        self.layers = nn.ModuleList(
            [
                BalancedAttentionLayer(
                    d_model=d_model,
                    d_head=d_head,
                    d_value=d_value,
                    sigma_qk=sigma_qk,
                    sigma_ov=sigma_ov,
                    seed=seed + 1000 * layer_idx,
                    lambda_qk=lambda_qk,
                    lambda_ov=lambda_ov,
                    **options,
                )
                for layer_idx in range(n_layers)
            ]
        )

        # This classifier is deliberately separate from the balanced OV circuit.
        self.classifier = nn.Linear(d_model, n_classes, **options)

    @torch.no_grad()
    def initialization_audit(self):
        return {
            f"layer_{idx + 1}": layer.initialization_audit()
            for idx, layer in enumerate(self.layers)
        }

    @torch.no_grad()
    def check_invariance_drift(self, detailed=False):
        if detailed:
            return {
                f"layer_{idx + 1}": layer.check_invariance_drift(detailed=True)
                for idx, layer in enumerate(self.layers)
            }
        return [layer.check_invariance_drift(detailed=False) for layer in self.layers]

    def forward(self, x, attention_mask=None):
        B, L = x.shape
        if L > self.max_seq_len:
            raise ValueError("Input exceeds max_seq_len.")

        if attention_mask is None:
            attention_mask = x.ne(self.pad_token_id)
        else:
            if attention_mask.shape != x.shape:
                raise ValueError("attention_mask must match input_ids.")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)

        positions = torch.arange(L, device=x.device).unsqueeze(0)
        valid = attention_mask.unsqueeze(-1)
        h = (self.token_emb(x) + self.pos_emb(positions)) * valid

        attentions = []
        for layer in self.layers:
            attn_out, attn_weights = layer(h, attention_mask)
            h = (h + attn_out) * valid
            attentions.append(attn_weights)

        count = valid.sum(dim=1).clamp_min(1)
        pooled = h.sum(dim=1) / count
        return self.classifier(pooled), attentions


LABEL_NAMES = ("hatespeech", "normal", "offensive")
DATA_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_safe(value):
    """Represent nonfinite floating-point values as null in JSON output."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_published_data(data_dir, local_files_only=False):
    """Load the source files used by the Hugging Face HateXplain builder."""
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    loaded, hashes = {}, {}

    for filename in ("dataset.json", "post_id_divisions.json"):
        path = directory / filename
        if not path.exists():
            if local_files_only:
                raise FileNotFoundError(f"Missing local dataset file: {path}")
            print(f"Downloading {filename} from the HateXplain repository...", flush=True)
            with urllib.request.urlopen(DATA_URL + filename, timeout=60) as response:
                raw = response.read()
            json.loads(raw)
            temporary = path.with_suffix(".json.partial")
            temporary.write_bytes(raw)
            temporary.replace(path)

        raw = path.read_bytes()
        loaded[filename] = json.loads(raw)
        hashes[filename] = hashlib.sha256(raw).hexdigest()

    records = loaded["dataset.json"]
    divisions = loaded["post_id_divisions.json"]
    validation_key = next(
        (key for key in ("val", "valid", "validation") if key in divisions), None
    )
    if validation_key is None:
        raise ValueError("The published split file has no validation split.")

    split_ids = {
        "train": divisions["train"],
        "validation": divisions[validation_key],
        "test": divisions["test"],
    }

    seen = set()
    for name, ids in split_ids.items():
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"Duplicate or overlapping IDs in the {name} split.")
        missing = set(ids).difference(records)
        if missing:
            raise ValueError(f"{len(missing)} {name} IDs are missing from dataset.json.")
        seen.update(ids)

    return records, split_ids, hashes


class HateXplainDataset(Dataset):
    def __init__(
        self,
        records,
        post_ids,
        tokenizer,
        max_len=64,
        label_tie_policy="skip",
    ):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for word_ids alignment.")

        self.samples = []
        self.stats = Counter()
        self.stats["source_examples"] = len(post_ids)

        for post_id in post_ids:
            item = records[post_id]
            annotators = item.get("annotators", [])
            labels = (
                annotators.get("label", [])
                if isinstance(annotators, dict)
                else [a["label"] for a in annotators]
            )
            labels = [
                LABEL_NAMES.index(label.lower()) if isinstance(label, str) else int(label)
                for label in labels
            ]

            if any(label not in range(3) for label in labels):
                raise ValueError(f"Unexpected label for post {post_id}.")
            if not labels:
                self.stats["skipped_no_labels"] += 1
                continue

            counts = Counter(labels)
            winners = sorted(
                label for label, count in counts.items() if count == max(counts.values())
            )
            if len(winners) > 1:
                self.stats["label_ties"] += 1
                if label_tie_policy == "error":
                    raise ValueError(f"Tied label votes for post {post_id}.")
                if label_tie_policy == "skip":
                    self.stats["skipped_label_ties"] += 1
                    continue

            label = winners[0]
            words = item["post_tokens"]
            if not words:
                self.stats["skipped_empty_posts"] += 1
                continue

            valid_rationales = []
            for rationale in item.get("rationales", []):
                if len(rationale) != len(words) or any(v not in (0, 1) for v in rationale):
                    self.stats["invalid_rationale_annotations"] += 1
                    continue
                valid_rationales.append(rationale)

            # Keep all valid rationale annotations. For rationale mass we want
            #
            #   (1/A) sum_a sum_j attention_j * rationale_j^(a),
            #
            # which is exactly
            #
            #   sum_j attention_j * mean_a[rationale_j^(a)].
            #
            # Therefore no hard consensus threshold is needed.
            n_rationales = len(valid_rationales)
            if n_rationales:
                mean_rationale = np.asarray(valid_rationales, dtype=np.float32).mean(axis=0)
                self.stats["examples_with_valid_annotations"] += 1
                self.stats["valid_rationale_annotations"] += n_rationales
            else:
                mean_rationale = np.zeros(len(words), dtype=np.float32)
                self.stats["examples_without_valid_annotations"] += 1

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )

            word_ids = encoding.word_ids(batch_index=0)
            attention_mask = encoding["attention_mask"].squeeze(0).bool()
            special_mask = encoding["special_tokens_mask"].squeeze(0).bool()
            input_ids = encoding["input_ids"].squeeze(0)

            content_mask = attention_mask & ~special_mask

            # Map the annotator-averaged word-level rationale weights to
            # WordPieces. Every WordPiece belonging to a word receives that
            # word's mean annotation weight. Multiplication by content_mask
            # excludes padding and inserted special tokens.
            rationale_weights = torch.tensor(
                [float(mean_rationale[w]) if w is not None else 0.0 for w in word_ids],
                dtype=torch.float32,
            ) * content_mask.float()

            has_rationale_annotations = n_rationales > 0
            has_retained_positive_rationale = bool((rationale_weights > 0).any())
            self.stats["examples_with_retained_positive_rationale"] += int(
                has_retained_positive_rationale
            )
            self.stats["positive_rationale_not_retained"] += int(
                bool(np.any(mean_rationale > 0)) and not has_retained_positive_rationale
            )

            self.samples.append(
                {
                    "post_id": str(post_id),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "special_tokens_mask": special_mask,
                    "rationale_weights": rationale_weights,
                    "has_rationale_annotations": torch.tensor(has_rationale_annotations),
                    "n_rationales": torch.tensor(n_rationales, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.long),
                }
            )
            self.stats[f"label_{LABEL_NAMES[label]}"] += 1

        self.stats["retained_examples"] = len(self.samples)
        if not self.samples:
            raise ValueError("No examples remain after preprocessing.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_loader(dataset, batch_size, shuffle, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def mean_attention_scores(attentions, attention_mask):
    """Average attention over layers and valid queries, including special keys."""
    if not attentions:
        raise ValueError("No attention matrices were returned by the model.")

    # Each entry is B x L x L. Averaging here is across layers only.
    attention = torch.stack(attentions, dim=0).mean(dim=0)
    valid = attention_mask.bool()
    scores = (attention * valid.unsqueeze(-1)).sum(dim=1)
    scores = scores / valid.sum(dim=1, keepdim=True).clamp_min(1)
    scores = scores * valid
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)


def select_top_content(scores, attention_mask, special_mask, fraction):
    content = attention_mask.bool() & ~special_mask.bool()
    count = content.sum(dim=1)
    k = torch.floor(count.float() * fraction).long().clamp_min(1)
    k = torch.minimum(k, count)
    order = scores.masked_fill(~content, -torch.inf).argsort(
        dim=-1, descending=True, stable=True
    )
    rank = torch.empty_like(order)
    rank.scatter_(
        1,
        order,
        torch.arange(scores.shape[1], device=scores.device).expand_as(order),
    )
    return content & (rank < k.unsqueeze(1)), count > 0


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    explanations=False,
    top_k_pct=0.20,
    faithfulness_target="gold",
    example_output=None,
):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    confusion = torch.zeros(3, 3, dtype=torch.long)
    mass_all_sum, mass_sum, mass_count = 0.0, 0.0, 0
    comp_sum, suff_sum, faith_count = 0.0, 0.0, 0
    example_rows = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, attentions = model(ids, mask)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite logits during evaluation.")

        predictions = logits.argmax(dim=1)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += predictions.eq(labels).sum().item()
        count += labels.numel()
        confusion += torch.bincount(
            (labels * 3 + predictions).cpu(), minlength=9
        ).view(3, 3)

        if not explanations:
            continue

        scores = mean_attention_scores(attentions, mask)

        # rationale_weights[j] = mean_a r_j^(a). Hence
        # (scores * rationale_weights).sum() is exactly the average, over all
        # valid rationale annotations for this example, of the attention mass
        # assigned to that annotation.
        rationale_weights = batch["rationale_weights"].to(device)
        eligible = batch["has_rationale_annotations"].to(device)
        n_rationales = batch["n_rationales"].to(device)
        mass = (scores * rationale_weights).sum(dim=1)
        mass_all_sum += mass.sum().item()
        mass_sum += mass[eligible].sum().item()
        mass_count += eligible.sum().item()

        special = batch["special_tokens_mask"].to(device)
        selected, has_content = select_top_content(scores, mask, special, top_k_pct)
        comp_mask = mask & ~selected
        suff_mask = mask & (selected | special)
        comp_ids = ids.masked_fill(~comp_mask, model.pad_token_id)
        suff_ids = ids.masked_fill(~suff_mask, model.pad_token_id)

        perturbed, _ = model(
            torch.cat((comp_ids, suff_ids)),
            torch.cat((comp_mask, suff_mask)),
        )
        if not torch.isfinite(perturbed).all():
            raise FloatingPointError("Nonfinite logits on perturbed inputs.")

        targets = labels if faithfulness_target == "gold" else predictions
        targets = targets.unsqueeze(1)
        p_orig = logits.softmax(dim=-1).gather(1, targets).squeeze(1)
        p_comp, p_suff = perturbed.softmax(dim=-1).chunk(2)
        comp = p_orig - p_comp.gather(1, targets).squeeze(1)
        suff = p_orig - p_suff.gather(1, targets).squeeze(1)

        comp_sum += comp[has_content].sum().item()
        suff_sum += suff[has_content].sum().item()
        faith_count += has_content.sum().item()

        if example_output is not None:
            for i, post_id in enumerate(batch["post_id"]):
                example_rows.append(
                    {
                        "post_id": post_id,
                        "label": int(labels[i]),
                        "prediction": int(predictions[i]),
                        "attention_mass": float(mass[i]),
                        "eligible_for_rationale_mass": bool(eligible[i]),
                        "n_rationales": int(n_rationales[i]),
                        "comprehensiveness": float(comp[i]) if has_content[i] else None,
                        "sufficiency": float(suff[i]) if has_content[i] else None,
                        "selected_wordpiece_positions": selected[i]
                        .nonzero()
                        .flatten()
                        .tolist(),
                    }
                )

    if count == 0:
        raise ValueError("Cannot evaluate an empty loader.")

    true_positive = confusion.diag().double()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = 2 * true_positive / denominator.clamp_min(1)

    result = {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "macro_f1": f1.mean().item(),
        "n_examples": count,
        "confusion_matrix": confusion.tolist(),
    }

    if explanations:
        result.update(
            {
                "rationale_mass": mass_sum / mass_count if mass_count else None,
                "rationale_mass_n": mass_count,
                "rationale_mass_all_examples": mass_all_sum / count,
                "comprehensiveness": comp_sum / faith_count if faith_count else None,
                "sufficiency": suff_sum / faith_count if faith_count else None,
                "faithfulness_n": faith_count,
            }
        )

    if example_output is not None:
        with Path(example_output).open("w") as stream:
            for row in example_rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    return result


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(ids, mask)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; inspect the learning rate.")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        count += labels.numel()

    return {"loss": total_loss / count, "accuracy": correct / count}


def _format_drift(drift):
    parts = []
    for layer_name, values in drift.items():
        # Report the direct balancedness residual ||B(t) - lambda I||_F.
        # The detailed JSON also retains ||B(t) - B(0)||_F as ``drift``.
        parts.append(
            f"{layer_name}: QK bal {values['qk']['current_target_error']:.3g}, "
            f"OV bal {values['ov']['current_target_error']:.3g}"
        )
    return " | ".join(parts)


def run_seed(args, seed, datasets, tokenizer, device):
    seed_everything(seed)
    run_dir = Path(args.output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    loaders = {
        name: make_loader(data, args.eval_batch_size, False, seed, device)
        for name, data in datasets.items()
    }
    train_loader = make_loader(datasets["train"], args.batch_size, True, seed, device)

    model_config = dict(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        d_head=args.d_head,
        d_value=args.d_value,
        n_layers=args.n_layers,
        n_classes=3,
        max_seq_len=args.max_len,
        lambda_qk=args.lambda_qk,
        lambda_ov=args.lambda_ov,
        sigma_qk=args.sigma_qk,
        sigma_ov=args.sigma_ov,
        seed=seed,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = BalancedTransformerClassifier(**model_config).to(device)

    initialization = model.initialization_audit()
    write_json(run_dir / "initialization.json", json_safe(initialization))
    for layer_name, layer_values in initialization.items():
        for circuit in ("qk", "ov"):
            values = layer_values[circuit]
            print(
                f"Seed {seed}, {layer_name}, {circuit.upper()}: "
                f"balance error {values['initial_target_error']:.6g}; "
                f"product error {values['effective_product_error']:.6g}",
                flush=True,
            )

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, loaders["validation"], device)
        drift = model.check_invariance_drift(detailed=True)

        row = {
            "epoch": epoch,
            "train_during_epoch": train_metrics,
            "validation": validation,
            "balance": json_safe(drift),
        }
        with (run_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "validation": validation,
            "training_config": vars(args),
        }

        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint, run_dir / "best.pt")
        if epoch == args.epochs:
            torch.save(checkpoint, run_dir / "final.pt")

        print(
            f"Seed {seed} | epoch {epoch}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {validation['loss']:.4f} | "
            f"val acc {validation['accuracy']:.4f} | "
            f"{_format_drift(drift)}",
            flush=True,
        )

    selected = torch.load(
        run_dir / f"{args.checkpoint}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(selected["model_state"])

    results = {
        "seed": seed,
        "checkpoint": args.checkpoint,
        "epoch": selected["epoch"],
        "balance": json_safe(model.check_invariance_drift(detailed=True)),
    }

    for split, loader in loaders.items():
        explain = split == "test" or args.explain_all_splits
        results[split] = evaluate(
            model,
            loader,
            device,
            explanations=explain,
            top_k_pct=args.top_k_pct,
            faithfulness_target=args.faithfulness_target,
            example_output=(
                run_dir / f"{split}_examples.jsonl" if explain else None
            ),
        )

    write_json(run_dir / "results.json", results)
    print(
        json.dumps(
            {
                "seed": seed,
                "evaluated_epoch": selected["epoch"],
                "test": results["test"],
            },
            indent=2,
        ),
        flush=True,
    )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="hatexplain_data")
    parser.add_argument("--output-dir", default="hatexplain_runs")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--d-head",
        type=int,
        default=None,
        help="Q/K latent dimension; defaults to d_model.",
    )
    parser.add_argument(
        "--d-value",
        type=int,
        default=None,
        help="V/O latent dimension; defaults to d_model.",
    )
    parser.add_argument("--n-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lambda-qk", type=float, default=200.0)
    parser.add_argument("--lambda-ov", type=float, default=40.0)
    parser.add_argument("--sigma-qk", type=float, default=0.1)
    parser.add_argument("--sigma-ov", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument(
        "--label-tie-policy", choices=("skip", "lowest", "error"), default="skip"
    )
    parser.add_argument("--top-k-pct", type=float, default=0.20)
    parser.add_argument(
        "--faithfulness-target",
        choices=("gold", "predicted"),
        default="predicted",
    )
    parser.add_argument("--explain-all-splits", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")

    args = parser.parse_args(argv)

    if args.d_head is None:
        args.d_head = args.d_model
    if args.d_value is None:
        args.d_value = args.d_model

    if min(
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.d_model,
        args.d_head,
        args.d_value,
    ) < 1:
        parser.error("Epochs, batch sizes, and model dimensions must be positive.")
    if args.d_head > args.d_model:
        parser.error("d_head must be <= d_model.")
    if args.d_value > args.d_model:
        parser.error("d_value must be <= d_model.")
    if args.max_len < 3:
        parser.error("max_len must be >= 3.")
    if not 0 < args.top_k_pct <= 1:
        parser.error("top_k_pct must be in (0,1].")

    finite_values = (
        args.lr,
        args.weight_decay,
        args.momentum,
        args.sigma_qk,
        args.sigma_ov,
        args.lambda_qk,
        args.lambda_ov,
    )
    if not all(math.isfinite(x) for x in finite_values):
        parser.error("Optimization, sigma, and lambda values must be finite.")
    if args.lr <= 0:
        parser.error("Learning rate must be positive.")
    if min(args.weight_decay, args.momentum, args.sigma_qk, args.sigma_ov) < 0:
        parser.error("Weight decay, momentum, and sigma values must be nonnegative.")
    if args.optimizer == "adamw" and args.momentum != 0:
        parser.error("--momentum only applies to SGD; pass --momentum 0 with AdamW.")
    if len(set(args.seeds)) != len(args.seeds) or any(
        s < 0 or s >= 2**32 for s in args.seeds
    ):
        parser.error("Seeds must be distinct integers between 0 and 2**32-1.")

    return args


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer, __version__ as transformers_version

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use CPU or CUDA; float64 initialization audits need support.")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() or any(
        (root / f"seed_{s}").exists() for s in args.seeds
    ):
        raise FileExistsError("Choose a new --output-dir to avoid overwriting an earlier run.")

    records, split_ids, hashes = load_published_data(
        args.data_dir, args.local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a PAD token.")

    datasets = {
        name: HateXplainDataset(
            records,
            ids,
            tokenizer,
            args.max_len,
            args.label_tie_policy,
        )
        for name, ids in split_ids.items()
    }

    print(
        json.dumps({name: dict(data.stats) for name, data in datasets.items()}, indent=2)
    )
    print(
        f"Device: {device}; layers: {args.n_layers}; d_model: {args.d_model}; "
        f"d_head: {args.d_head}; d_value: {args.d_value}; "
        f"optimizer: {args.optimizer}; lr: {args.lr}; "
        f"weight decay: {args.weight_decay}; checkpoint: {args.checkpoint}",
        flush=True,
    )

    if args.optimizer != "sgd" or args.weight_decay != 0 or args.momentum != 0:
        print(
            "Optimizer updates do not exactly conserve the continuous-time "
            "Gram differences; drift is recorded for every layer.",
            flush=True,
        )

    tokenizer.save_pretrained(root / "tokenizer")
    write_json(
        root / "config.json",
        {
            "arguments": vars(args),
            "dataset_sha256": hashes,
            "preprocessing": {name: dict(data.stats) for name, data in datasets.items()},
            "label_names": LABEL_NAMES,
            "torch_version": str(torch.__version__),
            "transformers_version": transformers_version,
            "numpy_version": str(np.__version__),
            "device": str(device),
        },
    )

    results = [
        run_seed(args, seed, datasets, tokenizer, device) for seed in args.seeds
    ]

    summary = {"seeds": args.seeds, "n_seeds": len(results), "test": {}}
    for metric in (
        "loss",
        "accuracy",
        "macro_f1",
        "rationale_mass",
        "rationale_mass_all_examples",
        "comprehensiveness",
        "sufficiency",
    ):
        values = [
            result["test"][metric]
            for result in results
            if result["test"][metric] is not None
        ]
        summary["test"][metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n_seeds": len(values),
        }

    write_json(root / "summary.json", summary)
    print("Final test summary:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
