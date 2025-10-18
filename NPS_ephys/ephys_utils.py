import re
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors



def process_df_state(df_state, states):
    # Rename the column
    df_state.rename(columns={"I_Injected(pA)": "Injected"}, inplace=True)

    # Create and order the 'Injected' categorical column
    current_cat = np.arange(25, 175, 10)
    df_state["Injected"] = pd.Categorical(
        df_state["Injected"], current_cat
    ).as_ordered()

    # Sort by 'Injected' and drop NaN values
    df_state.sort_values("Injected", inplace=True)
    df_state = df_state.dropna()

    # Create and order the 'state' categorical column
    df_state["state"] = pd.Categorical(df_state["state"], states).as_ordered()

    # Generate a DataFrame with all unique combinations of 'filename' and 'state'
    unique_combinations = df_state[["filename", "state"]].drop_duplicates()

    # Create a MultiIndex with all combinations of unique_combinations and current_cat
    all_combinations = (
        unique_combinations.assign(dummy=1)
        .merge(pd.DataFrame({"Injected": current_cat, "dummy": 1}), on="dummy")
        .drop("dummy", axis=1)
    )

    # Merge with the original data to ensure all combinations are present
    df_state_full = all_combinations.merge(
        df_state, on=["filename", "Injected", "state"], how="left"
    )
    # Calculate the number of spikes
    num_spike = (
        df_state.groupby(by=["filename", "Injected", "state"])["Event start "]
        .nunique()
        .reset_index(name="num_spike")
    )

    # Merge all combinations with num_spike to ensure all categories are present
    num_spike = all_combinations.merge(
        num_spike, on=["filename", "Injected", "state"], how="left"
    ).fillna({"num_spike": 0})

    # Get the first occurrence in each group and drop duplicates
    df_all = (
        df_state_full.groupby(["filename", "Injected", "state"])
        .first()
        .reset_index()
        .drop_duplicates()
    )

    # Calculate the latency
    latency = (
        df_state.groupby(["filename", "Injected", "state"])["Event start "]
        .min()
        .reset_index(name="latency_event_start")
    )

    # Merge the calculated values into df_all
    df_all = pd.merge(df_all, num_spike, on=["filename", "Injected", "state"])
    df_all = pd.merge(df_all, latency, on=["filename", "Injected", "state"])
    df_all["latency"] = df_all["latency_event_start"] - df_all["Ramp_start"]

    return df_all


def test_interaction_pvalues(
    sampled_df, prop, combos, state_col="state", group_var="filename", correction=True
):
    p_values = pd.DataFrame(columns=["comparison", "interaction", "pvalue"])

    for combo in combos:
        sub_df = sampled_df[sampled_df[state_col].isin(combo)].dropna(subset = 'filename')
        sub_df[state_col] = sub_df[state_col].astype("object")


        model = smf.mixedlm(
            f"{prop} ~ Injected * {state_col}",
            sub_df,
            groups=sub_df[group_var],
        ).fit()

        # Find all interaction terms containing both 'Injected' and 'state'
        interaction_terms = [
            term
            for term in model.pvalues.index
            if "Injected" in term and state_col in term
        ]

        for interaction_term in interaction_terms:
            pvalue = model.pvalues[interaction_term]
            match = re.search(r"Injected\[T\.([^\]]+)\]", interaction_term)
            injected = match.group(1) if match else "unknown"
            p_values = pd.concat(
                [
                    p_values,
                    pd.DataFrame(
                        {
                            "comparison": [f"{combo[0]} vs {combo[1]}"],
                            "interaction": [interaction_term],
                            "injected": [injected],
                            "pvalue": [pvalue],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # Apply multiple testing correction if requested
    if correction and not p_values.empty:
        reject, pvals_corrected, _, _ = multipletests(
            p_values["pvalue"], alpha=0.05, method="fdr_bh"
        )
        p_values["pvalue_corrected"] = pvals_corrected
        p_values["reject_null"] = reject

    return p_values


def plot_significance_texts_shifted(
    ax, sampled_df, result_df, combos, palette, state_col="state", base_y=0, y_shift=0.2
):
    """
    Plot significance stars near y=0 with slight vertical shifts and combo-specific colors.

    Parameters:
    - ax: matplotlib axis
    - sampled_df: DataFrame with 'state', 'Injected', prop columns
    - result_df: DataFrame with ['comparison', 'interaction', 'pvalue_corrected', 'reject_null']
    - combos: list of tuples of state pairs, e.g. [("Pre", "Agg+"), ...]
    - state_col: str, column name for states in sampled_df
    - palette: dict or list of colors corresponding to states or combos
    - base_y: float, baseline y coordinate to start placing stars
    - y_shift: float, vertical distance between stars for different combos
    """

    # Map combos to colors (assuming palette keys are states)
    # For combos, average the colors of two states:
    import matplotlib.colors as mcolors

    def avg_color(c1, c2):
        c1_rgb = np.array(mcolors.to_rgb(c1))
        c2_rgb = np.array(mcolors.to_rgb(c2))
        avg_rgb = (c1_rgb + c2_rgb) / 2
        return avg_rgb

    combo_colors = {}
    for c in combos:
        if isinstance(palette, dict):
            c1 = palette[c[0]]
            c2 = palette[c[1]]
        else:
            # If palette is list, map states to indices first (assumes states order matches palette)
            # You can adjust this as needed.
            states = sampled_df[state_col].unique().tolist()
            c1 = palette[states.index(c[0])]
            c2 = palette[states.index(c[1])]
        combo_colors[c] = avg_color(c1, c2)

    signif_results = result_df[result_df["reject_null"]]

    for i, combo in enumerate(combos):
        # Filter results for this combo
        mask = signif_results["comparison"] == f"{combo[0]} vs {combo[1]}"
        combo_results = signif_results[mask]

        for _, row in combo_results.iterrows():
            pval = row["pvalue_corrected"]
            term = row["interaction"]
            match = re.search(r"Injected\[T\.([^\]]+)\]", term)
            if not match:
                continue
            injected_val = float(match.group(1))

            # Calculate y position (shifted by i*y_shift above base_y)
            y = base_y + (i - 1) * y_shift

            if pval < 0.001:
                text = "***"
            elif pval < 0.01:
                text = "**"
            elif pval < 0.05:
                text = "*"
            else:
                text = f"p={pval:.2f}"

            ax.text(
                injected_val,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=13,
                color=combo_colors[combo],
            )

def plot_significance_bars_shifted(
    ax, sampled_df, result_df, combos, palette, state_col="state", base_y=0, y_shift=0.2, bar_height=0.5, bar_width=10
):
    """
    Plot significance indicators as semi-transparent bars near y=0 with vertical shifts and combo-specific colors.
    
    Parameters:
    - ax: matplotlib axis
    - sampled_df: DataFrame with 'state', 'Injected', prop columns
    - result_df: DataFrame with ['comparison', 'interaction', 'pvalue_corrected', 'reject_null']
    - combos: list of tuples of state pairs, e.g. [("Pre", "Agg+"), ...]
    - state_col: str, column name for states in sampled_df
    - palette: dict or list of colors corresponding to states or combos
    - base_y: float, baseline y coordinate to start placing bars
    - y_shift: float, vertical distance between bars for different combos
    - bar_height: float, height of each significance bar
    - bar_width: float, width of the bar in x-axis
    """

    def avg_color(c1, c2):
        c1_rgb = np.array(mcolors.to_rgb(c1))
        c2_rgb = np.array(mcolors.to_rgb(c2))
        return (c1_rgb + c2_rgb) / 2

    # Get colors for each combo
    combo_colors = {}
    for c in combos:
        if isinstance(palette, dict):
            c1, c2 = palette[c[0]], palette[c[1]]
        else:
            states = sampled_df[state_col].unique().tolist()
            c1, c2 = palette[states.index(c[0])], palette[states.index(c[1])]
        combo_colors[c] = avg_color(c1, c2)

    signif_results = result_df[result_df["reject_null"]]

    for i, combo in enumerate(combos):
        combo_mask = signif_results["comparison"] == f"{combo[0]} vs {combo[1]}"
        combo_results = signif_results[combo_mask]

        for _, row in combo_results.iterrows():
            pval = row["pvalue_corrected"]
            term = row["interaction"]
            match = re.search(r"Injected\[T\.([^\]]+)\]", term)
         
            if not match:
                continue
            injected_val = float(match.group(1))
            y = base_y + (i - 1) * y_shift

            # Determine alpha from significance level
            if pval < 0.001:
                alpha = 0.8
            elif pval < 0.01:
                alpha = 0.5
            elif pval < 0.05:
                alpha = 0.2
            else:
                continue  # skip not significant

            y = base_y + (i - 1) * y_shift
            color = combo_colors[combo]

            rect = Rectangle(
                (injected_val - bar_width / 2, y),
                width=bar_width,
                height=bar_height,
                color=color,
                alpha=alpha,
                linewidth=0
            )
         
            ax.add_patch(rect)
            rect.set_zorder(10)  # After add_patch

    # if there are no rectangle, don't add legend
    if  ax.patches:
    
        legend_patches = [
            mpatches.Patch(color='grey', alpha=0.8, label='p < 0.001'),
            mpatches.Patch(color='grey', alpha=0.5, label='p < -0.01'),
            mpatches.Patch(color='grey', alpha=0.2, label='p < 0.05'),
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=10, bbox_to_anchor=(0, 1))
        leg = ax.get_legend()
        if leg:
            leg.set_frame_on(False)
