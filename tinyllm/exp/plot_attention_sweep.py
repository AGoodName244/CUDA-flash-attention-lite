import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("attention_sweep_results.csv")
df_ok = df[df["status"] == "ok"].copy()

IMPLS = ["naive", "ext", "online"]


def plot_prefill_time(df):
    prefill = df[df["tag"] == "prefill"].copy()
    prefill = prefill.sort_values("Tkv")

    plt.figure()
    for impl in IMPLS:
        sub = prefill[prefill["impl"] == impl]
        if sub.empty:
            continue
        plt.plot(sub["Tkv"], sub["time_ms"], marker="o", label=impl)

    plt.xlabel("Sequence length T (Tq = Tkv)")
    plt.ylabel("Time per iter (ms)")
    plt.title("Prefill: Time vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prefill_time_vs_T.png")


def plot_prefill_time_log(df):
    prefill = df[df["tag"] == "prefill"].copy()
    prefill = prefill.sort_values("Tkv")

    plt.figure()
    for impl in IMPLS:
        sub = prefill[prefill["impl"] == impl]
        if sub.empty:
            continue
        plt.plot(sub["Tkv"], sub["time_ms"], marker="o", label=impl)

    plt.xlabel("Sequence length T (Tq = Tkv)")
    plt.ylabel("Time per iter (ms, log scale)")
    plt.title("Prefill: Time vs Sequence Length (log y)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("prefill_time_vs_T_log.png")


def plot_prefill_memory(df):
    prefill = df[df["tag"] == "prefill"].copy()
    prefill = prefill.sort_values("Tkv")

    plt.figure()
    for impl in IMPLS:
        sub = prefill[prefill["impl"] == impl]
        if sub.empty:
            continue
        plt.plot(sub["Tkv"], sub["peak_mem_MB"], marker="o", label=impl)

    plt.xlabel("Sequence length T (Tq = Tkv)")
    plt.ylabel("Peak CUDA memory (MB)")
    plt.title("Prefill: Peak Memory vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prefill_mem_vs_T.png")


def plot_prefill_tflops(df):
    prefill = df[df["tag"] == "prefill"].copy()
    prefill = prefill.sort_values("Tkv")

    plt.figure()
    for impl in IMPLS:
        sub = prefill[prefill["impl"] == impl]
        if sub.empty or "tflops" not in sub.columns:
            continue
        plt.plot(sub["Tkv"], sub["tflops"], marker="o", label=impl)

    plt.xlabel("Sequence length T (Tq = Tkv)")
    plt.ylabel("Effective TFLOP/s")
    plt.title("Prefill: TFLOP/s vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prefill_tflops_vs_T.png")


def plot_decode_time(df):
    decode = df[df["tag"] == "decode"].copy()
    decode = decode.sort_values("Tkv")

    plt.figure()
    for impl in IMPLS:
        sub = decode[decode["impl"] == impl]
        if sub.empty:
            continue
        plt.plot(sub["Tkv"], sub["time_ms"], marker="o", label=impl)

    plt.xlabel("Cache length Tkv (Tq = 1)")
    plt.ylabel("Time per iter (ms)")
    plt.title("Decode: Time vs Cache Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("decode_time_vs_Tkv.png")


def plot_decode_tflops(df):
    decode = df[df["tag"] == "decode"].copy()
    decode = decode.sort_values("Tkv")

    plt.figure()
    for impl in IMPLS:
        sub = decode[decode["impl"] == impl]
        if sub.empty or "tflops" not in sub.columns:
            continue
        plt.plot(sub["Tkv"], sub["tflops"], marker="o", label=impl)

    plt.xlabel("Cache length Tkv (Tq = 1)")
    plt.ylabel("Effective TFLOP/s")
    plt.title("Decode: TFLOP/s vs Cache Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("decode_tflops_vs_Tkv.png")


def plot_D_sweep_time(df):
    ds = df[df["tag"] == "D_sweep"].copy()
    ds = ds.sort_values("D")

    plt.figure()
    for impl in IMPLS:
        sub = ds[ds["impl"] == impl]
        if sub.empty:
            continue
        plt.plot(sub["D"], sub["time_ms"], marker="o", label=impl)

    plt.xlabel("Head dimension D")
    plt.ylabel("Time per iter (ms)")
    plt.title("Time vs Head Dimension (D_sweep)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("D_sweep_time_vs_D.png")


def plot_D_sweep_tflops(df):
    ds = df[df["tag"] == "D_sweep"].copy()
    ds = ds.sort_values("D")

    plt.figure()
    for impl in IMPLS:
        sub = ds[ds["impl"] == impl]
        if sub.empty or "tflops" not in sub.columns:
            continue
        plt.plot(sub["D"], sub["tflops"], marker="o", label=impl)

    plt.xlabel("Head dimension D")
    plt.ylabel("Effective TFLOP/s")
    plt.title("TFLOP/s vs Head Dimension (D_sweep)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("D_sweep_tflops_vs_D.png")


if __name__ == "__main__":
    # prefill: time & log-time & mem & tflops
    plot_prefill_time(df_ok)
    plot_prefill_time_log(df_ok)
    plot_prefill_memory(df_ok)
    plot_prefill_tflops(df_ok)

    # decode: time & tflops
    plot_decode_time(df_ok)
    plot_decode_tflops(df_ok)

    # D sweep: time & tflops
    plot_D_sweep_time(df_ok)
    plot_D_sweep_tflops(df_ok)

    print("All plots saved.")
