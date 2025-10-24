
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="CURIOUS Animal Traits", layout="wide")

st.title("🐘🐘CURIOUS Animal Traits🐘🐘")
st.caption("Authors: James Cleaver, Lauren McKnight, Maria Pettyjohn")

st.write("""In this activity, you’ll use **data science** to explore patterns in a large dataset (big data) and create **data visualisations** to help you make sense of what you see.

You’ll be exploring an open database of **land-dwelling animals**, compiled from thousands of scientific papers. You can read more abou this dataset at animaltraits.org

NOTE: The *common name* column was added to the animaltraits.org dataset by automatic searching of other tables. This column may include some errors.

Starting with something familiar — **animal body size** — you’ll learn how scientists use data to spot patterns and relationships.

Along the way, you’ll practise thinking like a data scientist: organising and exploring data, choosing effective scales, and visualising relationships to explain what the data tells us about the natural world.

The sidebar works as your digital worksheet to keep track of your observations. The traffic light emoji 🚦 will point out when to answer a sidebar question.

Let's get started!
""")



# --- Configuration ---
DEFAULT_CSV = "observations2.csv"

# --- Load data automatically ---
if Path(DEFAULT_CSV).exists():
    try:
        data = pd.read_csv(DEFAULT_CSV)
    except Exception as e:
        st.error(f" Could not read {DEFAULT_CSV}: {e}")
        st.stop()
else:
    st.error(f"⚠️ Default CSV '{DEFAULT_CSV}' not found in the app folder.")
    st.stop()

# --- Sidebar with guiding questions ---
st.sidebar.title("🧠 Investigation Guide")

st.sidebar.markdown("Use this sidebar to record your thinking as you explore the data. Note that answers will not be saved if the page closes or refreshes")

questions = {
    "q1": "Q1- What does each row in the dataset represent? What columns (or variables) are included?",
    "q2": "Q2- Search for your favourite animal. Write down its species name and the 'class' it is in. ",
    "q3": "Q3- What is the biggest animal in the dataset? What is the smallest?",
    "q4": "Q4- Compare the linear and log scales. Which one helped you understand the distribution of body masses better? Why? (What does that tell you about how animal body sizes are spread across the dataset?)",
    "q5": "Q5- Which type of graph helped you compare the animal classes most clearly?",
    "q6": "Q6- Describe the relationship between body size and brain size using the template *as body mass increases/decreases, brain size tends to increase/decrease* or by describing the correlation mathematically."
}

responses = {}
for key, question in questions.items():
    responses[key] = st.sidebar.text_area(question, height=80)

# --- Main page ---
st.title("Dataset Viewer")
st.header("A) Data preview")
st.write("The dataset is displayed in the table below")
st.caption(f"Dataset: **{DEFAULT_CSV}**  |  Rows: {len(data):,}  |  Columns: {len(data.columns)}")
st.dataframe(data, use_container_width=True)
st.write("If you hover over the table, a magnifying glass will appear top right that you can use to search for your favourite animal.")
st.write("🚦 Answer questions 1-2 in the sidebar")
# --- Section: Explore a single variable ---
st.header("B) Exploring One Variable: Body Mass (kg)")
st.write("""
A **variable** is something that can vary or change between animals — like body mass, brain size, or metabolic rate (how much energy they use while resting).
In this dataset, each column represents a variable describing some aspect of the animals.
""")

st.write("🚦 Using the table above, click the column heading to sort the data by **body mass (kg)**, then answer Question 3 in the sidebar.")

# --- Brief data visualisation explanation ---
st.subheader("Understanding Data Visualisation")
st.write("""
**Data visualisation** means creating graphs or charts based on a dataset. This can help us to see patterns and trends in the data more easily than looking at a table of numbers.
A **histogram** is a type of graph that shows how frequently different values occur — for example, how many animals fall into each weight range.
""")

# --- Histogram Section ---
st.subheader("Plot a Histogram")

variable_option = st.selectbox(
    "Choose a variable to plot:",
    options=["body mass (kg)", "brain size (kg)"],
    index=0
)
buckets_option = st.number_input(
    "Enter the number (between 5 and 100) of buckets (bins) for the histogram:",
    min_value=5,
    max_value=100,
    value=25
)
scale_option = st.radio(
    "Choose the scale for the x-axis:",
    options=["Linear", "Logarithmic"],
    horizontal=True,
    key="scale_histogram"
)

# Labels
labels_list = {
    "body mass (kg)": "Body Mass (kilograms)",
    "brain size (kg)": "Brain Size (kilograms)"
}
label = labels_list.get(variable_option, variable_option)

# Prepare data
data_all = data[variable_option].dropna()
data_all = data_all[data_all > 0]  # remove nonpositive values

# Bin calculation
if scale_option == "Linear":
    bins = np.linspace(data_all.min(), data_all.max(), int(buckets_option) + 1)
else:
    lower = max(data_all.min() * 0.8, 1e-9)
    upper = data_all.max() * 1.2
    bins = np.logspace(np.log10(lower), np.log10(upper), int(buckets_option) + 1)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(
    data_all,
    bins=bins,
    color="skyblue",
    edgecolor="black",
    alpha=0.8,
)
if scale_option == "Logarithmic":
    ax.set_xscale("log")

ax.set_xlabel(label)
ax.set_ylabel("Number of Animals")
ax.set_title(f"{scale_option} Distribution of {label}")
ax.tick_params(axis="both", labelsize=9)
fig.tight_layout()

st.pyplot(fig, use_container_width=False)


# --- Tip and reflection ---
st.markdown("""
💡 A *logarithmic scale* spaces numbers according to their **powers of ten** rather than evenly.  
This makes very large and very small values easier to see together — for example, animals that differ by a thousand times in mass.

🚦 Try using different scales for body mass to find the most helpful visualisation. Then answer Question 4 in the sidebar.
""")

# --- Section: Compare Animal Classes ---
st.header("C) Comparing Animal Classes")

st.write("""
Now let’s compare how **body mass** differs between different groups of animals.

Each *class* represents a broad group, such as mammals, birds, or reptiles.
Different graphs help us see how each group’s data are distributed and whether some groups tend to be heavier or lighter.
""")

# Map scientific to common names
class_labels = {
    "Amphibia": "Amphibians",
    "Arachnida": "Spiders & Scorpions",
    "Aves": "Birds",
    "Insecta": "Insects",
    "Malacostraca": "Crustaceans",
    "Mammalia": "Mammals",
    "Clitellata": "Worms",
    "Gastropoda": "Snails & Slugs",
    "Reptilia": "Reptiles"
}

# Prepare data
plot_data = data.copy()
plot_data["Class (common name)"] = plot_data["class"].map(class_labels)
plot_data = plot_data.dropna(subset=["body mass (kg)", "Class (common name)"])
plot_data = plot_data[plot_data["body mass (kg)"] > 0]

classes = plot_data["Class (common name)"].unique()
positions = np.arange(len(classes))
colours = plt.cm.tab10(np.linspace(0, 1, len(classes)))  # distinct palette

# Graph type selection with friendly display labels and defaults
graph_display_labels = {
    "Box and Whisker": "Box and Whisker Plot",
    "Violin": "Violin Plot",
    "Average ± Error Bars": "Average ± Error Bars",
    "Strip (Jitter)": "Individual Points"
}

# The internal values (for your plotting code)
graph_options = list(graph_display_labels.keys())

# Create the selectbox, defaulting to "Strip (Jitter)"
selected_label = st.selectbox(
    "Choose a graph type:",
    options=[graph_display_labels[g] for g in graph_options],
    index=graph_options.index("Strip (Jitter)")  # default
)

# Reverse-map to the internal value for your logic
graph_type = [k for k, v in graph_display_labels.items() if v == selected_label][0]

scale_option = st.radio(
    "Choose y-axis scale:",
    ["Linear", "Logarithmic"],
    horizontal=True,
    key="scale_class_compare",
    index=1  # ✅ default to "Logarithmic"
)

fig, ax = plt.subplots(figsize=(8, 5))

# --- Box and Whisker ---
if graph_type == "Box and Whisker":
    box_data = [
        plot_data.loc[plot_data["Class (common name)"] == cls, "body mass (kg)"]
        for cls in classes
    ]
    boxprops = dict(linewidth=1.0, color="black")
    flierprops = dict(marker="x", markersize=3, color="black", alpha=0.6)
    ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=flierprops,
    )
    for patch, color in zip(ax.artists, colours):
        patch.set_facecolor(color)
    ax.set_title("Body Mass by Animal Class (Box and Whisker)")

# --- Violin ---
elif graph_type == "Violin":
    violin_data = [
        plot_data.loc[plot_data["Class (common name)"] == cls, "body mass (kg)"]
        for cls in classes
    ]
    parts = ax.violinplot(
        violin_data,
        positions=positions,
        showmedians=True,
        widths=0.8
    )
    for pc, c in zip(parts['bodies'], colours):
        pc.set_facecolor(c)
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)
    parts["cmedians"].set_color("black")
    ax.set_title("Body Mass by Animal Class (Violin Plot)")

# --- Average ± Error Bars ---
elif graph_type == "Average ± Error Bars":
    means = [plot_data.loc[plot_data["Class (common name)"] == cls, "body mass (kg)"].mean() for cls in classes]
    stds = [plot_data.loc[plot_data["Class (common name)"] == cls, "body mass (kg)"].std() for cls in classes]
    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt='o',
        ecolor='black',
        elinewidth=1,
        capsize=4,
        markersize=6,
        markerfacecolor='white',
        markeredgecolor='black',
        color='black'
    )
    for i, (m, s, c) in enumerate(zip(means, stds, colours)):
        ax.plot(positions[i], m, 'o', color=c, markersize=8)
    ax.set_title("Average Body Mass by Animal Class (Mean ± SD)")

# --- Strip (Jitter) ---
else:
    for i, (cls, color) in enumerate(zip(classes, colours)):
        y_vals = plot_data.loc[plot_data["Class (common name)"] == cls, "body mass (kg)"]
        jitter_x = np.random.normal(i, 0.08, len(y_vals))
        ax.scatter(jitter_x, y_vals, alpha=0.6, s=10, color=color, label=cls)
    ax.set_title("Body Mass by Animal Class (Strip Plot)")

# --- Formatting ---
ax.set_xticks(positions)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_xlabel("Animal Class")
ax.set_ylabel("Body Mass (kg)")
ax.set_ylim(1e-9, 1e3)  # consistent with histogram section
if scale_option == "Logarithmic":
    ax.set_yscale("log")

fig.tight_layout()
st.pyplot(fig, use_container_width=True)





# --- Reflection question ---
st.markdown("""
🚦 Try different graph types. Which do you like the best? Answer question 5 in the sidebar.
""")


# --- SECTION D: Explore Relationships Between Variables ---

st.header("D) Explore Relationships Between Variables")

st.write("""
In this final section, you'll look for **relationships between two traits** — for example, whether animals
with larger body masses also tend to have larger brains or higher metabolic rates. 
""")

# ------------------------------------------------------------------
# 🟩 PART 1 – DEMO GRAPH (fixed, colourful, hoverable)
# ------------------------------------------------------------------

st.subheader("Demo: Relationship between Body Mass and Brain Size")

st.write("""
This graph is a *scatter plot* showing the relationship between two variables.  
The patterns we can see are called *correlations* — this means the variables are "related together" - there is a connection between these traits.  
Looking for these relationships is one of the important ways scientists discover how the world works. 

Each point on the graph represents a different species.  
Both axes use **log scales**, which makes it possible to compare very small and very large animals on the same chart.  

In this demo, the animal classes are shown in different colours.  
Move your mouse over the points to see each species’ **common name** and class.  
Notice the overall pattern: even though animals vary enormously in size, the relationship between
body mass and brain size forms a nearly straight line — with slightly different slopes for each class.
""")


demo_df = data.copy()
demo_df.columns = demo_df.columns.str.strip().str.replace('\u00a0', ' ', regex=True)
demo_df["Class (common name)"] = demo_df["class"].map({
    "Amphibia": "Amphibians",
    "Arachnida": "Spiders & Scorpions",
    "Aves": "Birds",
    "Insecta": "Insects",
    "Malacostraca": "Crustaceans",
    "Mammalia": "Mammals",
    "Clitellata": "Worms",
    "Gastropoda": "Snails & Slugs",
    "Reptilia": "Reptiles",
})
demo_df = demo_df.dropna(subset=["body mass (kg)", "brain size (kg)"])
demo_df = demo_df[(demo_df["body mass (kg)"] > 0) & (demo_df["brain size (kg)"] > 0)]

fig_demo = px.scatter(
    demo_df,
    x="body mass (kg)",
    y="brain size (kg)",
    color="Class (common name)",
    hover_data=["common name"],
    log_x=True,
    log_y=True,
    title="Body Mass vs Brain Size (log–log scale)",
    height=550,
)
fig_demo.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0)))
fig_demo.update_layout(legend_title_text="Animal Class", margin=dict(l=10, r=10, t=40, b=10))

st.plotly_chart(fig_demo, use_container_width=True)

st.markdown("🚦 What overall pattern do you notice between body mass and brain size across all animals? Answer question 6 in the sidebar")
st.markdown(""" 
            💬 **Advanced:** Can you find humans? What does their position mean about the mass of the human brain in relation to our body size?
            
            💬 Extra advanced: Why are their multiple dots for humans? What does each represent? (You may need to look back at the source of data to find this answer)
            """)

st.divider()

# ------------------------------------------------------------------
# 🟦 PART 2 – INTERACTIVE GRAPH (editable version with optional line)
# ------------------------------------------------------------------

st.subheader("D2) Try it yourself!")

st.write("""
In the graph below you can choose different variables to explore.
""")

st.write("""
Use the dropdowns to choose two variables. 
All species are shown in grey; you can **highlight** one animal class. Optional: turn on the **line of best fit** to see the slope/equation. (Advanced note: the slope is really interesting to scientists! look up "power laws in scaling" if you want to know more!)
""")

# Clean column names once (keeps things robust if CSV headings had odd spaces)
data.columns = data.columns.str.strip().str.replace('\u00a0', ' ', regex=True)

# Map classes to readable names (omit Chilopoda)
class_map = {
    "Amphibia": "Amphibians",
    "Arachnida": "Spiders & Scorpions",
    "Aves": "Birds",
    "Insecta": "Insects",
    "Malacostraca": "Crustaceans",
    "Mammalia": "Mammals",
    "Clitellata": "Worms",
    "Gastropoda": "Snails & Slugs",
    "Reptilia": "Reptiles",
}
plot_data = data.copy()
plot_data["Class (common name)"] = plot_data["class"].map(class_map)

variables = [
    "body mass (kg)",
    "metabolic rate (W)",
    "mass-specific metabolic rate (W/kg)",
    "brain size (kg)",
]

# --- Default selections ---
default_x = "body mass (kg)"
default_y = "brain size (kg)"
default_highlight = "Mammals"

# --- Layout of selection boxes ---
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    x_var = st.selectbox("X-axis variable:", options=variables, index=variables.index(default_x))
with c2:
    y_var = st.selectbox("Y-axis variable:", options=[v for v in variables if v != x_var],
                         index=[v for v in variables if v != x_var].index(default_y) if default_y in variables else 0)
with c3:
    highlight_class = st.selectbox(
        "Highlight one class (optional):",
        options=["All"] + sorted(plot_data["Class (common name)"].dropna().unique().tolist()),
        index=(["All"] + sorted(plot_data["Class (common name)"].dropna().unique().tolist())).index(default_highlight)
    )

# --- Line of best fit unchecked by default ---
show_fit = st.checkbox("Show line of best fit for the highlighted class", value=False)

# --- Prepare subset safely for log–log ---
subset = plot_data[[x_var, y_var, "Class (common name)"]].copy()
subset[x_var] = pd.to_numeric(subset[x_var], errors="coerce")
subset[y_var] = pd.to_numeric(subset[y_var], errors="coerce")
subset = subset.dropna(subset=[x_var, y_var])
subset = subset[(subset[x_var] > 0) & (subset[y_var] > 0)]

if subset.empty:
    st.warning(f"No positive values available for {x_var} and {y_var} after cleaning.")
else:
    fig, ax = plt.subplots(figsize=(7, 5))

    # All species (grey)
    ax.scatter(
        subset[x_var], subset[y_var],
        s=25, color="lightgrey", alpha=0.6, edgecolor="none", label="All species"
    )

    # Optional highlight overlay
    if highlight_class != "All":
        hi = subset[subset["Class (common name)"] == highlight_class]
        if not hi.empty:
            ax.scatter(
                hi[x_var], hi[y_var],
                s=40, color="tab:red", alpha=0.9, edgecolor="black", linewidth=0.3,
                label=highlight_class
            )

            if show_fit and len(hi) > 2:
                # Fit in log10 space so slope is the scaling exponent
                x_log = np.log10(hi[x_var].values)
                y_log = np.log10(hi[y_var].values)
                m, b = np.polyfit(x_log, y_log, 1)  # y_log = m*x_log + b

                x_line = np.linspace(x_log.min(), x_log.max(), 200)
                y_line = m * x_line + b
                ax.plot(10**x_line, 10**y_line, color="tab:red", linewidth=2.0, label=f"{highlight_class} best fit")

                # Show equation + slope
                st.markdown(
                    f"""
                    🔹 **Best-fit line (log–log):**  
                    `log10({y_var}) = {m:.3f} × log10({x_var}) + {b:.3f}`  
                    **Equivalent power-law:**  
                    `{y_var} ≈ 10^{b:.3f} × ({x_var})^{m:.3f}`  
                    _(Slope / exponent = {m:.3f})_
                    """
                )

    # Axes + styling (always log–log)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"{y_var} vs {x_var} (Log–Log)")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    st.caption(f"Showing {len(subset):,} valid points (positive values only).")

# --- Reflection ---
st.markdown("""


💬 **Advanced:**  
When plotted on a log–log scale, straight lines show *scaling laws* — one quantity changing in proportion to another.  
The **slope** tells us *how fast* one grows compared to the other.  

💬 What does the slope you found suggest about how these traits scale across species? Is it the same for different classes of animal?

💬 Extra advanced: why might the correlation of metabolic rate and body size follow a cube-square law?
""")


