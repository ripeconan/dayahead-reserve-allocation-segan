"""R32a-fix: load full SAOCP score-stream method bodies from pinned source AST.

Production path is a single fail-closed gate:

    load_saocp_ast_kernel(repo)

requires a pristine Git checkout of the pinned commit (clean worktree,
commit/tree/blob/path/bytes/sha256) **before any compile/exec**.

Compile always revalidates the *actual* AST nodes that will be compiled
(deep-copy + ``ast.dump`` SHA vs frozen map). Callers cannot inject a
forged identity or mutated ``ExtractedNodeSet`` past the private compiler.

Test-only offline fixtures use ``load_saocp_ast_kernel_for_test`` and never
claim verified git commit/tree/clean. Independent synthetic reference lives
only in the test module (not here).

Backend kind is never ``official_direct``. Production registry is not wired.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import subprocess
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - pandas is a hard dep for SAOCP update
    raise ImportError(
        "pandas is required to load the SAOCP AST kernel (official update uses pd.Series)"
    ) from exc


# ---------------------------------------------------------------------------
# Frozen pinned source identity (R03 / R10 / verified_wrapper contract)
# ---------------------------------------------------------------------------

PINNED_SAOCP_SOURCE_IDENTITY: dict[str, Any] = {
    "repo_label": (
        "/home/icdm/card10_external/card10_r12_method_lock/clones/saocp_online_conformal"
    ),
    "commit": "95790a9b9e4061166e6d7a8a13282e019b1a8663",
    "tree": "77348e5fe7f2bb3ed5a031aaad9874b6a5ab2d7b",
    "files": {
        "online_conformal/saocp.py": {
            "sha256": "83f8845559786452aafabfafdf6c136ceaf9b9b3557fc5f8f9267fcec04d386f",
            "bytes": 5175,
            "git_blob": "0b44d7e3afadc114bd49fc83653c6afcf8738a46",
        },
        "online_conformal/utils.py": {
            "sha256": "88c8e7725f114ccd5c68ebd4adcd6804c523aa241fa62f2b29c2a95354d83fa1",
            "bytes": 3263,
            "git_blob": "66d0c4e8a532de1096369de078b1060841e3337b",
        },
    },
}

# Deterministic ast.dump SHA256 for the eight required nodes on the pinned bytes.
# Re-validated by extract/load/compile; tests re-parse source and compare.
PINNED_SAOCP_NODE_HASHES: dict[str, str] = {
    "saocp:_OGD": (
        "f1b5a7a29f434e5890990e71a64486cab906b1856d221969d5ba620693eda3fa"
    ),
    "saocp:SAOCP.get_p": (
        "67b4ef88cc76e086b9a0876b9e903ea0f74d86883dc0c6bac2d8ca82f3f5c0b6"
    ),
    "saocp:SAOCP.predict": (
        "a2039b625939081d30bfa6dfaf6e22a72b31f5fa3144b4aacdc9528a0d1bb97c"
    ),
    "saocp:SAOCP.create_expert": (
        "6d03ef053e7a6eb5ed3b29dfcb12ecc3f4b138acaa226f298859c183fccf8203"
    ),
    "saocp:SAOCP.update": (
        "e2c512a6431bc75acce8a684f43c8f8d4d03de6c19d9011913dc8ff3abbc95b8"
    ),
    "utils:pinball_loss": (
        "9d6f2269379fd74403d12808fc03a781af68a9a469bd9f3491602ba3e999034f"
    ),
    "utils:pinball_loss_grad": (
        "f7f5db530f64b0914f1160e91dea4f4d64ce9bb1322978c1112e48d059a46844"
    ),
    "utils:Residuals": (
        "d14111b1de92a5ffc8f61bf005a72b6984a2fa46eb1c483d45554ef13c6cff1c"
    ),
}

SAOCP_METHOD_NAMES = ("get_p", "predict", "create_expert", "update")
REQUIRED_NODE_KEYS = tuple(PINNED_SAOCP_NODE_HASHES.keys())

_BACKEND_KIND = "verified_wrapper_ast_kernel"  # never official_direct

# Internal gate token type for private compile path. Not forgeable from outside
# without going through _freeze_nodes_for_compile (which re-hashes real ASTs).
_COMPILE_GATE_SECRET = object()


class SaocpAstKernelError(RuntimeError):
    """Fail-closed error before or during pinned SAOCP AST kernel load."""


@dataclass(frozen=True)
class _CompileGate:
    """Opaque proof that freeze/revalidation succeeded for a node set."""

    secret: object
    node_hashes: tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# Hash / git helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def node_dump_sha256(node: ast.AST) -> str:
    """SHA256 of ``ast.dump(node, include_attributes=False)`` (UTF-8)."""
    dumped = ast.dump(node, include_attributes=False)
    return sha256_bytes(dumped.encode("utf-8"))


def _run_git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SaocpAstKernelError(f"git invocation failed in {repo}: {exc!r}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SaocpAstKernelError(
            f"git {' '.join(args)} failed in {repo} (rc={proc.returncode}): {err}"
        )
    return (proc.stdout or "").strip()


def _read_pinned_file(repo: Path, relpath: str) -> tuple[Path, bytes, str]:
    path = repo / relpath
    if not path.is_file():
        raise SaocpAstKernelError(f"pinned source missing: {path}")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    expected = PINNED_SAOCP_SOURCE_IDENTITY["files"][relpath]
    if len(data) != int(expected["bytes"]):
        raise SaocpAstKernelError(
            f"source byte length mismatch for {relpath}: "
            f"got {len(data)}, expected {expected['bytes']}"
        )
    if digest != expected["sha256"]:
        raise SaocpAstKernelError(
            f"source sha256 mismatch for {relpath}: got {digest}, "
            f"expected {expected['sha256']}"
        )
    return path, data, digest


def _verify_pinned_saocp_files_for_test(repo: Path) -> dict[str, Any]:
    """Test-only file path/bytes/sha256 check. Never claims git verification.

    Identity uses ``expected_commit`` / ``expected_tree`` (not bare verified
    ``commit`` / ``tree``) and marks all git_*_verified flags false.
    """
    repo = Path(repo)
    files_meta: dict[str, Any] = {}
    for relpath, expected in PINNED_SAOCP_SOURCE_IDENTITY["files"].items():
        path, data, digest = _read_pinned_file(repo, relpath)
        files_meta[relpath] = {
            "path": str(path.resolve()),
            "bytes": len(data),
            "sha256": digest,
            "git_blob_expected": expected["git_blob"],
        }
    return {
        "identity_mode": "test_only_file_hash",
        "test_only": True,
        "git_commit_verified": False,
        "git_tree_verified": False,
        "clean_verified": False,
        "expected_commit": PINNED_SAOCP_SOURCE_IDENTITY["commit"],
        "expected_tree": PINNED_SAOCP_SOURCE_IDENTITY["tree"],
        "files": files_meta,
        "repo": str(repo.resolve()),
    }


def verify_pinned_saocp_repo(repo: Path) -> dict[str, Any]:
    """Fail-fast identity gate: clean worktree, commit, tree, blobs, file hashes.

    Clean is always required (no ``require_clean=False`` escape hatch).
    Must be called before any compile/exec of official method bodies.
    """
    repo = Path(repo)
    if not repo.is_dir():
        raise SaocpAstKernelError(f"SAOCP repo path is not a directory: {repo}")
    if not (repo / ".git").exists():
        raise SaocpAstKernelError(
            f"SAOCP repo is not a git checkout (missing .git): {repo}"
        )

    commit = _run_git(repo, "rev-parse", "HEAD")
    tree = _run_git(repo, "rev-parse", "HEAD^{tree}")
    expected_commit = PINNED_SAOCP_SOURCE_IDENTITY["commit"]
    expected_tree = PINNED_SAOCP_SOURCE_IDENTITY["tree"]
    if commit != expected_commit:
        raise SaocpAstKernelError(
            f"commit mismatch: got {commit}, expected {expected_commit}"
        )
    if tree != expected_tree:
        raise SaocpAstKernelError(
            f"tree mismatch: got {tree}, expected {expected_tree}"
        )

    porcelain = _run_git(repo, "status", "--porcelain")
    if porcelain:
        raise SaocpAstKernelError(
            f"repo is dirty (refuse load before exec):\n{porcelain}"
        )

    files_meta: dict[str, Any] = {}
    for relpath, expected in PINNED_SAOCP_SOURCE_IDENTITY["files"].items():
        path, data, digest = _read_pinned_file(repo, relpath)
        blob = _run_git(repo, "ls-tree", "HEAD", relpath).split()
        # format: <mode> blob <sha>\t<path>  or space-separated
        blob_sha = None
        if len(blob) >= 3 and blob[1] == "blob":
            blob_sha = blob[2]
        if blob_sha != expected["git_blob"]:
            raise SaocpAstKernelError(
                f"git blob mismatch for {relpath}: got {blob_sha}, "
                f"expected {expected['git_blob']}"
            )
        files_meta[relpath] = {
            "path": str(path.resolve()),
            "bytes": len(data),
            "sha256": digest,
            "git_blob": blob_sha,
        }

    return {
        "identity_mode": "git_full",
        "test_only": False,
        "git_commit_verified": True,
        "git_tree_verified": True,
        "clean_verified": True,
        "repo": str(repo.resolve()),
        "repo_label": PINNED_SAOCP_SOURCE_IDENTITY["repo_label"],
        "commit": commit,
        "tree": tree,
        "dirty": False,
        "files": files_meta,
    }


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------


@dataclass
class ExtractedNode:
    key: str
    qualname: str
    source_relpath: str
    node: ast.AST
    node_sha256: str


@dataclass
class ExtractedNodeSet:
    nodes: dict[str, ExtractedNode]
    saocp_sha256: str
    utils_sha256: str
    saocp_bytes: int
    utils_bytes: int

    def hashes(self) -> dict[str, str]:
        return {k: v.node_sha256 for k, v in self.nodes.items()}


def _find_unique(
    items: list[ast.AST],
    predicate: Callable[[ast.AST], bool],
    label: str,
) -> ast.AST:
    matches = [n for n in items if predicate(n)]
    if not matches:
        raise SaocpAstKernelError(f"required AST node not found: {label}")
    if len(matches) > 1:
        raise SaocpAstKernelError(
            f"duplicate AST nodes for {label}: count={len(matches)}"
        )
    return matches[0]


def extract_saocp_ast_nodes(saocp_src: str, utils_src: str) -> ExtractedNodeSet:
    """Parse vendor sources and extract exactly the eight required AST nodes.

    Does not compile or exec. Fail-closed on missing/duplicate nodes or when
    recomputed node hashes disagree with the frozen pinned map.
    """
    if not isinstance(saocp_src, str) or not isinstance(utils_src, str):
        raise SaocpAstKernelError("sources must be str (decoded UTF-8 text)")

    saocp_b = saocp_src.encode("utf-8")
    utils_b = utils_src.encode("utf-8")
    # Byte-identity gate even when caller bypassed file path checks.
    exp_s = PINNED_SAOCP_SOURCE_IDENTITY["files"]["online_conformal/saocp.py"]
    exp_u = PINNED_SAOCP_SOURCE_IDENTITY["files"]["online_conformal/utils.py"]
    if len(saocp_b) != exp_s["bytes"] or sha256_bytes(saocp_b) != exp_s["sha256"]:
        raise SaocpAstKernelError(
            "saocp.py source bytes/sha256 do not match pinned identity "
            "(refuse extract/exec)"
        )
    if len(utils_b) != exp_u["bytes"] or sha256_bytes(utils_b) != exp_u["sha256"]:
        raise SaocpAstKernelError(
            "utils.py source bytes/sha256 do not match pinned identity "
            "(refuse extract/exec)"
        )

    try:
        saocp_tree = ast.parse(saocp_src)
        utils_tree = ast.parse(utils_src)
    except SyntaxError as exc:
        raise SaocpAstKernelError(f"failed to parse pinned source: {exc}") from exc

    saocp_body = list(saocp_tree.body)
    utils_body = list(utils_tree.body)

    ogd = _find_unique(
        saocp_body,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "_OGD",
        "class _OGD",
    )
    saocp_cls = _find_unique(
        saocp_body,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "SAOCP",
        "class SAOCP",
    )
    assert isinstance(saocp_cls, ast.ClassDef)

    method_nodes: dict[str, ast.FunctionDef] = {}
    for name in SAOCP_METHOD_NAMES:
        matches = [
            n
            for n in saocp_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if not matches:
            raise SaocpAstKernelError(f"SAOCP method not found: {name}")
        if len(matches) > 1:
            raise SaocpAstKernelError(
                f"duplicate SAOCP method definitions: {name} count={len(matches)}"
            )
        method_nodes[name] = matches[0]

    pinball_loss_node = _find_unique(
        utils_body,
        lambda n: isinstance(n, ast.FunctionDef) and n.name == "pinball_loss",
        "function pinball_loss",
    )
    pinball_loss_grad_node = _find_unique(
        utils_body,
        lambda n: isinstance(n, ast.FunctionDef) and n.name == "pinball_loss_grad",
        "function pinball_loss_grad",
    )
    residuals_node = _find_unique(
        utils_body,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "Residuals",
        "class Residuals",
    )

    raw: list[tuple[str, str, str, ast.AST]] = [
        ("saocp:_OGD", "_OGD", "online_conformal/saocp.py", ogd),
        (
            "saocp:SAOCP.get_p",
            "SAOCP.get_p",
            "online_conformal/saocp.py",
            method_nodes["get_p"],
        ),
        (
            "saocp:SAOCP.predict",
            "SAOCP.predict",
            "online_conformal/saocp.py",
            method_nodes["predict"],
        ),
        (
            "saocp:SAOCP.create_expert",
            "SAOCP.create_expert",
            "online_conformal/saocp.py",
            method_nodes["create_expert"],
        ),
        (
            "saocp:SAOCP.update",
            "SAOCP.update",
            "online_conformal/saocp.py",
            method_nodes["update"],
        ),
        (
            "utils:pinball_loss",
            "pinball_loss",
            "online_conformal/utils.py",
            pinball_loss_node,
        ),
        (
            "utils:pinball_loss_grad",
            "pinball_loss_grad",
            "online_conformal/utils.py",
            pinball_loss_grad_node,
        ),
        (
            "utils:Residuals",
            "Residuals",
            "online_conformal/utils.py",
            residuals_node,
        ),
    ]

    nodes: dict[str, ExtractedNode] = {}
    for key, qualname, relpath, node in raw:
        h = node_dump_sha256(node)
        expected_h = PINNED_SAOCP_NODE_HASHES[key]
        if h != expected_h:
            raise SaocpAstKernelError(
                f"node hash mismatch for {key}: got {h}, expected {expected_h}"
            )
        nodes[key] = ExtractedNode(
            key=key,
            qualname=qualname,
            source_relpath=relpath,
            node=node,
            node_sha256=h,
        )

    if set(nodes) != set(REQUIRED_NODE_KEYS):
        raise SaocpAstKernelError(
            f"node key set mismatch: got {sorted(nodes)}, "
            f"expected {list(REQUIRED_NODE_KEYS)}"
        )

    return ExtractedNodeSet(
        nodes=nodes,
        saocp_sha256=sha256_bytes(saocp_b),
        utils_sha256=sha256_bytes(utils_b),
        saocp_bytes=len(saocp_b),
        utils_bytes=len(utils_b),
    )


def extract_saocp_ast_nodes_unchecked(
    saocp_src: str, utils_src: str
) -> dict[str, ast.AST]:
    """Extract the eight nodes without pinned byte/hash gates.

    Intended for fail-closed unit tests that deliberately mutate source. Still
    refuses missing/duplicate required names. Does **not** compile/exec.
    """
    saocp_tree = ast.parse(saocp_src)
    utils_tree = ast.parse(utils_src)
    saocp_body = list(saocp_tree.body)
    utils_body = list(utils_tree.body)

    ogd = _find_unique(
        saocp_body,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "_OGD",
        "class _OGD",
    )
    saocp_cls = _find_unique(
        saocp_body,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "SAOCP",
        "class SAOCP",
    )
    assert isinstance(saocp_cls, ast.ClassDef)
    methods: dict[str, ast.AST] = {}
    for name in SAOCP_METHOD_NAMES:
        matches = [
            n
            for n in saocp_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if not matches:
            raise SaocpAstKernelError(f"SAOCP method not found: {name}")
        if len(matches) > 1:
            raise SaocpAstKernelError(
                f"duplicate SAOCP method definitions: {name} count={len(matches)}"
            )
        methods[name] = matches[0]

    out: dict[str, ast.AST] = {
        "saocp:_OGD": ogd,
        "saocp:SAOCP.get_p": methods["get_p"],
        "saocp:SAOCP.predict": methods["predict"],
        "saocp:SAOCP.create_expert": methods["create_expert"],
        "saocp:SAOCP.update": methods["update"],
        "utils:pinball_loss": _find_unique(
            utils_body,
            lambda n: isinstance(n, ast.FunctionDef) and n.name == "pinball_loss",
            "function pinball_loss",
        ),
        "utils:pinball_loss_grad": _find_unique(
            utils_body,
            lambda n: isinstance(n, ast.FunctionDef) and n.name == "pinball_loss_grad",
            "function pinball_loss_grad",
        ),
        "utils:Residuals": _find_unique(
            utils_body,
            lambda n: isinstance(n, ast.ClassDef) and n.name == "Residuals",
            "class Residuals",
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Compile / bind kernel (private production helper)
# ---------------------------------------------------------------------------


def _source_label(relpath: str, qualname: str, node_sha256: str) -> str:
    commit = PINNED_SAOCP_SOURCE_IDENTITY["commit"]
    tree = PINNED_SAOCP_SOURCE_IDENTITY["tree"]
    return (
        f"<card10_saocp_ast:commit={commit}:tree={tree}:"
        f"{relpath}:{qualname}:node_sha256={node_sha256}>"
    )


def _exec_node(node: ast.AST, ns: dict[str, Any], filename: str) -> None:
    """Compile and exec a single AST class/function node into ``ns``."""
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        code = compile(module, filename, "exec")
    except Exception as exc:
        raise SaocpAstKernelError(
            f"compile failed for {filename}: {exc!r}"
        ) from exc
    exec(code, ns, ns)  # noqa: S102 — intentional, pinned AST only


@dataclass
class SaocpAstKernelBundle:
    """Loaded kernel class + official symbols from pinned AST nodes."""

    kernel_cls: type
    OGD: type
    Residuals: type
    pinball_loss: Callable[..., Any]
    pinball_loss_grad: Callable[..., Any]
    node_hashes: dict[str, str]
    identity: dict[str, Any]
    source_labels: dict[str, str] = field(default_factory=dict)
    backend_kind: str = _BACKEND_KIND

    def method_code_meta(self) -> dict[str, dict[str, str]]:
        """Map kernel method name -> co_filename / node hash for audit."""
        out: dict[str, dict[str, str]] = {}
        for name in SAOCP_METHOD_NAMES:
            fn = getattr(self.kernel_cls, name)
            code = fn.__code__
            key = f"saocp:SAOCP.{name}"
            out[name] = {
                "co_filename": code.co_filename,
                "node_key": key,
                "node_sha256": self.node_hashes[key],
            }
        return out


def _freeze_nodes_for_compile(
    extracted: ExtractedNodeSet,
) -> tuple[dict[str, ExtractedNode], _CompileGate]:
    """Revalidate metadata + deep-copy AST; rehash copies against frozen map.

    compile/exec may only consume the returned deep copies. Any post-extract
    mutation of constants/operators/bodies, forged ``node_sha256`` fields, or
    source-byte metadata drift fails here **before** the first exec.
    """
    exp_s = PINNED_SAOCP_SOURCE_IDENTITY["files"]["online_conformal/saocp.py"]
    exp_u = PINNED_SAOCP_SOURCE_IDENTITY["files"]["online_conformal/utils.py"]
    if (
        int(extracted.saocp_bytes) != int(exp_s["bytes"])
        or extracted.saocp_sha256 != exp_s["sha256"]
    ):
        raise SaocpAstKernelError(
            "pre-compile saocp source metadata mismatch "
            f"(bytes={extracted.saocp_bytes}, sha256={extracted.saocp_sha256})"
        )
    if (
        int(extracted.utils_bytes) != int(exp_u["bytes"])
        or extracted.utils_sha256 != exp_u["sha256"]
    ):
        raise SaocpAstKernelError(
            "pre-compile utils source metadata mismatch "
            f"(bytes={extracted.utils_bytes}, sha256={extracted.utils_sha256})"
        )

    if set(extracted.nodes) != set(REQUIRED_NODE_KEYS):
        raise SaocpAstKernelError(
            f"pre-compile node key set mismatch: got {sorted(extracted.nodes)}, "
            f"expected {list(REQUIRED_NODE_KEYS)}"
        )

    frozen: dict[str, ExtractedNode] = {}
    hash_pairs: list[tuple[str, str]] = []
    for key in REQUIRED_NODE_KEYS:
        en = extracted.nodes[key]
        node_copy = copy.deepcopy(en.node)
        recomputed = node_dump_sha256(node_copy)
        # Field must match the *actual* AST about to be compiled.
        if en.node_sha256 != recomputed:
            raise SaocpAstKernelError(
                f"pre-compile node_sha256 field mismatch for {key}: "
                f"field={en.node_sha256}, recomputed={recomputed}"
            )
        expected_h = PINNED_SAOCP_NODE_HASHES[key]
        if recomputed != expected_h:
            raise SaocpAstKernelError(
                f"pre-compile node hash mismatch for {key}: "
                f"got {recomputed}, expected {expected_h}"
            )
        frozen[key] = ExtractedNode(
            key=en.key,
            qualname=en.qualname,
            source_relpath=en.source_relpath,
            node=node_copy,
            node_sha256=recomputed,
        )
        hash_pairs.append((key, recomputed))

    gate = _CompileGate(
        secret=_COMPILE_GATE_SECRET,
        node_hashes=tuple(hash_pairs),
    )
    return frozen, gate


def _assert_compile_gate(gate: _CompileGate, frozen: Mapping[str, ExtractedNode]) -> None:
    if not isinstance(gate, _CompileGate) or gate.secret is not _COMPILE_GATE_SECRET:
        raise SaocpAstKernelError(
            "compile refused: invalid or forged compile gate token"
        )
    expected = dict(gate.node_hashes)
    if set(expected) != set(REQUIRED_NODE_KEYS) or set(frozen) != set(REQUIRED_NODE_KEYS):
        raise SaocpAstKernelError("compile refused: gate/node key set mismatch")
    for key in REQUIRED_NODE_KEYS:
        if node_dump_sha256(frozen[key].node) != expected[key]:
            raise SaocpAstKernelError(
                f"compile refused: frozen node drift for {key}"
            )
        if frozen[key].node_sha256 != expected[key]:
            raise SaocpAstKernelError(
                f"compile refused: frozen node_sha256 drift for {key}"
            )
        if expected[key] != PINNED_SAOCP_NODE_HASHES[key]:
            raise SaocpAstKernelError(
                f"compile refused: gate hash not pinned for {key}"
            )


def _compile_saocp_ast_kernel(
    extracted: ExtractedNodeSet,
    identity: Mapping[str, Any],
) -> SaocpAstKernelBundle:
    """Private compiler: freeze/revalidate then compile only deep-copied nodes.

    Not re-exported from ``load_official``. Callers cannot skip revalidation by
    supplying a forged ``identity``; identity is recorded only, not trusted as a
    compile gate.
    """
    frozen, gate = _freeze_nodes_for_compile(extracted)
    _assert_compile_gate(gate, frozen)

    from typing import Dict, Tuple  # local for annotation evaluation

    ns: dict[str, Any] = {
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "Dict": Dict,
        "Tuple": Tuple,
    }
    labels: dict[str, str] = {}

    # 1) utils functions / Residuals
    for key in ("utils:pinball_loss", "utils:pinball_loss_grad", "utils:Residuals"):
        en = frozen[key]
        label = _source_label(en.source_relpath, en.qualname, en.node_sha256)
        labels[key] = label
        _exec_node(en.node, ns, label)

    if "pinball_loss" not in ns or "pinball_loss_grad" not in ns or "Residuals" not in ns:
        raise SaocpAstKernelError("utils symbols missing after AST exec")

    # 2) _OGD (needs pinball_loss / pinball_loss_grad / np)
    en_ogd = frozen["saocp:_OGD"]
    label_ogd = _source_label(
        en_ogd.source_relpath, en_ogd.qualname, en_ogd.node_sha256
    )
    labels["saocp:_OGD"] = label_ogd
    _exec_node(en_ogd.node, ns, label_ogd)
    if "_OGD" not in ns:
        raise SaocpAstKernelError("_OGD missing after AST exec")

    # 3) SAOCP methods as free functions, then bind onto dynamic class
    method_objs: dict[str, Any] = {}
    for name in SAOCP_METHOD_NAMES:
        key = f"saocp:SAOCP.{name}"
        en = frozen[key]
        label = _source_label(en.source_relpath, en.qualname, en.node_sha256)
        labels[key] = label
        # Exec into a fresh local dict so we only pick up this function name,
        # but share globals for pinball_loss / _OGD / np lookups.
        local: dict[str, Any] = {}
        module = ast.Module(body=[en.node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, label, "exec")
        exec(code, ns, local)  # noqa: S102
        if name not in local:
            raise SaocpAstKernelError(f"method {name} missing after AST exec")
        method_objs[name] = local[name]
        # Also publish on shared ns so methods can see each other if needed
        # via globals (official bodies use self.* for sibling methods).
        ns[name] = local[name]

    # Dynamic kernel class — methods are official code objects; no handwritten bodies.
    kernel_cls = types.new_class(
        "SAOCPAstKernel",
        (),
        {},
        lambda ns_cls: ns_cls.update(
            {
                "__module__": "card10_baselines.saocp_ast_kernel",
                "__doc__": (
                    "Minimal state shell bound to pinned SAOCP AST method bodies. "
                    "backend_kind=verified_wrapper_ast_kernel (not official_direct)."
                ),
                "backend_kind": _BACKEND_KIND,
                **method_objs,
            }
        ),
    )

    node_hashes = {k: v.node_sha256 for k, v in frozen.items()}
    return SaocpAstKernelBundle(
        kernel_cls=kernel_cls,
        OGD=ns["_OGD"],
        Residuals=ns["Residuals"],
        pinball_loss=ns["pinball_loss"],
        pinball_loss_grad=ns["pinball_loss_grad"],
        node_hashes=node_hashes,
        identity=dict(identity),
        source_labels=labels,
        backend_kind=_BACKEND_KIND,
    )


def load_saocp_ast_kernel(repo: Path) -> SaocpAstKernelBundle:
    """Load the full SAOCP AST kernel from a pristine pinned Git checkout.

    Unconditionally verifies: path is a Git checkout, worktree is clean,
    commit/tree match the freeze, and both source blobs match path/bytes/SHA.
    Any mismatch fails **before** any compile/exec.

    No ``identity_mode`` / ``require_clean`` parameters and no env fallback.
    Offline fixtures must use ``load_saocp_ast_kernel_for_test``.
    """
    repo = Path(repo)
    identity = verify_pinned_saocp_repo(repo)

    # Re-read after identity gate (no exec yet).
    saocp_path = repo / "online_conformal" / "saocp.py"
    utils_path = repo / "online_conformal" / "utils.py"
    saocp_src = saocp_path.read_text(encoding="utf-8")
    utils_src = utils_path.read_text(encoding="utf-8")
    extracted = extract_saocp_ast_nodes(saocp_src, utils_src)
    return _compile_saocp_ast_kernel(extracted, identity)


def load_saocp_ast_kernel_for_test(repo: Path) -> SaocpAstKernelBundle:
    """Test-only offline fixture loader (file path/bytes/sha256 only).

    Identity explicitly marks ``test_only=true`` and all git verification
    flags false. Records ``expected_commit`` / ``expected_tree`` only — never
    bare verified ``commit`` / ``tree``. Not re-exported from ``load_official``.
    """
    repo = Path(repo)
    identity = _verify_pinned_saocp_files_for_test(repo)
    saocp_path = repo / "online_conformal" / "saocp.py"
    utils_path = repo / "online_conformal" / "utils.py"
    saocp_src = saocp_path.read_text(encoding="utf-8")
    utils_src = utils_path.read_text(encoding="utf-8")
    extracted = extract_saocp_ast_nodes(saocp_src, utils_src)
    return _compile_saocp_ast_kernel(extracted, identity)


def make_minimal_saocp_kernel_state(
    bundle: SaocpAstKernelBundle,
    *,
    coverage: float = 0.9,
    lifetime: int = 8,
    horizon: int = 1,
    scale: float = 1.0,
    t: int = 1,
) -> Any:
    """Adapter-only factory: empty object with fields official methods need.

    Does **not** call BasePredictor / public SAOCP ``__init__``. Not wired into
    production backends in R32a/R32a-fix.
    """
    obj = bundle.kernel_cls.__new__(bundle.kernel_cls)
    obj.t = int(t)
    obj.horizon = int(horizon)
    obj.lifetime = int(lifetime)
    obj.coverage = float(coverage)
    obj.scale = {h + 1: float(scale) for h in range(horizon)}
    obj.experts = {h + 1: {} for h in range(horizon)}
    obj.residuals = bundle.Residuals(horizon)
    return obj
