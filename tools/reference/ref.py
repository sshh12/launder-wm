"""Numpy reference of the SynthID g-value pipeline (HF transformers / synthid-text v0.2.1 variant)
plus the synthid-text @main variant. Verifies int64 semantics for a TS/BigInt port."""
import numpy as np, hashlib

np.seterr(over='ignore')

MULT = np.int64(6364136223846793005)
INC = np.int64(1)
M64 = (1 << 64)

def acc_hash_np(cur, data):
    """cur: int64 scalar/array; data: int64 array, last dim iterated."""
    cur = np.int64(cur)
    for i in range(data.shape[-1]):
        cur = cur + data[..., i]
        cur = cur * MULT
        cur = cur + INC
    return cur

def acc_hash_py(cur, data):
    """Pure-python 64-bit two's complement emulation (what BigInt TS must do)."""
    def wrap(x):
        x &= (M64 - 1)
        return x - M64 if x >= (1 << 63) else x
    for d in data:
        cur = wrap(cur + d)
        cur = wrap(cur * 6364136223846793005)
        cur = wrap(cur + 1)
    return cur

# ---- cross-check numpy int64 wrap == python two's complement wrap
rng = np.random.default_rng(0)
for _ in range(2000):
    data = rng.integers(0, 300000, size=5).astype(np.int64)
    iv = int(rng.integers(-2**62, 2**62))
    a = int(acc_hash_np(np.int64(iv), data))
    b = acc_hash_py(iv, [int(x) for x in data])
    assert a == b, (a, b, iv, data)
print("OK: numpy int64 accumulate_hash == python two's-complement wrap emulation")

# ---- semantics probes
x = np.int64(-12345678901234567)
print("arith >>30 of negative int64:", int(x >> np.int64(30)), " (python >> gives", (-12345678901234567) >> 30, ")")
print("numpy negative % 65536 :", int(x % np.int64(65536)), " (JS BigInt %% would give", -((-int(x)) % 65536), ")")
print("numpy negative % 2     :", int(x % np.int64(2)))

# ---- HF / v0.2.1 pipeline
def sampling_table_stub(size, seed):
    # NOTE: real table comes from torch.randint(0,2,(size,), generator=manual_seed(seed))
    # this stub is only for structural test vectors.
    r = np.random.default_rng(seed)
    return r.integers(0, 2, size=size).astype(np.int64)

class HFStyle:
    def __init__(self, ngram_len, keys, table_size=2**16, table_seed=0, ctx_hist=1024):
        self.H = ngram_len
        self.keys = np.array(keys, dtype=np.int64)
        self.table = sampling_table_stub(table_size, table_seed)
        self.table_size = table_size
        self.ctx_hist = ctx_hist

    def ngram_key(self, ngram):
        """ngram: list of ngram_len token ids -> per-depth int64 keys"""
        h = acc_hash_np(np.int64(1), np.array(ngram, dtype=np.int64))   # IV = 1
        return np.array([acc_hash_np(h, np.array([k], dtype=np.int64)) for k in self.keys], dtype=np.int64)

    def g_values(self, ids):
        out = []
        for i in range(len(ids) - self.H + 1):
            nk = self.ngram_key(ids[i:i+self.H])
            idx = nk % np.int64(self.table_size)   # numpy/torch remainder -> non-negative
            out.append(self.table[idx])
        return np.array(out)

    def context_hash(self, ctx):
        return int(acc_hash_np(np.int64(1), np.array(ctx, dtype=np.int64)))

    def repetition_mask(self, ids):
        hist = [0] * self.ctx_hist
        mask = []
        for i in range(len(ids) - self.H + 1):
            ctx = ids[i:i+self.H-1]
            ch = self.context_hash(ctx)
            mask.append(0 if ch in hist else 1)
            hist = [ch] + hist[:-1]
        return np.array(mask)

class MainStyle(HFStyle):
    def __init__(self, ngram_len, keys, ctx_hist=1024):
        self.H = ngram_len
        self.keys = np.array(keys, dtype=np.int64)
        self.ctx_hist = ctx_hist
        d = hashlib.sha256(self.keys.tobytes()).digest()
        self.hash_iv = int.from_bytes(d, byteorder='big') % (2**63 - 1)

    def ngram_key(self, ngram):
        h = acc_hash_np(np.int64(self.hash_iv), np.array(ngram, dtype=np.int64))
        return np.array([acc_hash_np(h, np.array([k], dtype=np.int64)) for k in self.keys], dtype=np.int64)

    def context_hash(self, ctx):
        return int(acc_hash_np(np.int64(self.hash_iv), np.array(ctx, dtype=np.int64)))

    def get_gvals(self, nk, num_apply_hash=12, shift=0):
        shift = shift or (64 // num_apply_hash)   # = 5
        k = nk.copy()
        for _ in range(num_apply_hash):
            k = acc_hash_np(k, np.array([1], dtype=np.int64).reshape(1,)[None].repeat(1,0)[0:1]) if False else \
                (acc_hash_np(k, np.array([1], dtype=np.int64)) >> np.int64(shift))
        return (k >> np.int64(30)) % np.int64(2)

    def g_values(self, ids):
        return np.array([self.get_gvals(self.ngram_key(ids[i:i+self.H]))
                         for i in range(len(ids) - self.H + 1)])

KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990,
        966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960]

hf = HFStyle(5, KEYS)
mn = MainStyle(5, KEYS)
print("\nmain-variant hash_iv =", mn.hash_iv, hex(mn.hash_iv))

toks = [1, 235280, 2121, 576, 573, 2121, 576, 573, 12, 99, 1, 235280, 2121, 576, 573]
print("\nRAW ngram keys (IV=1, HF/v0.2.1) for ngram", toks[:5], "->", [int(v) for v in hf.ngram_key(toks[:5])][:5])
print("RAW ngram keys (main, sha-IV)     for ngram", toks[:5], "->", [int(v) for v in mn.ngram_key(toks[:5])][:5])
print("\ncontext hashes (IV=1) :", [hf.context_hash(toks[i:i+4]) for i in range(4)])
print("repetition mask (HF)  :", hf.repetition_mask(toks).tolist())
print("main g-values depth0..4 per position:")
print(mn.g_values(toks)[:, :5])
