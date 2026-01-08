import glob, json, pandas as pd

rows = []
for fp in glob.glob("R6_stack_TbCp_stage3_inter/fold_*/meta_coefficients.json"):
    d = json.load(open(fp, "r"))
    c = d["coef_dict"]
    fold = d["fold"]

    def eff(m):
        base = c.get(m, 0.0)
        return {
            "I_II": base,
            "IIIa": base + c.get(f"{m}*stage3_IIIa", 0.0),
            "IIIb+": base + c.get(f"{m}*stage3_IIIbplus", 0.0),
        }

    tb = eff("Tb")
    cp = eff("Cp")
    rows.append({
        "fold": fold,
        "Tb_I_II": tb["I_II"], "Tb_IIIa": tb["IIIa"], "Tb_IIIb+": tb["IIIb+"],
        "Cp_I_II": cp["I_II"], "Cp_IIIa": cp["IIIa"], "Cp_IIIb+": cp["IIIb+"],
        "alpha": d["alpha"],
    })

df = pd.DataFrame(rows).sort_values("fold")
print(df)
print("\nMean:\n", df.mean(numeric_only=True))
