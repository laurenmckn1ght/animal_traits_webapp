
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

st.sidebar.markdown("Use this sidebar to record your thinking as you explore the data. You can view your answers at the end, but they will not be saved if the page closes or refreshes")

questions = {
    "q1": "Q1- What does each row in the dataset represent? What columns (or variables) are included?",
    "q2": "Q2- Search for your favourite animal. Write down its species name and the 'class' it is in. ",
    "q3": "Q3- What is the biggest animal in the dataset? What is the smallest?",
    "q4": "Q4- Compare the linear and log scales. Which one helped you understand the distribution of body masses better? Why? (What does that tell you about how animal body sizes are spread across the dataset?)",
    "q5": "Q5- Which type of graph helped you compare the animal classes most clearly?"
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

graph_type = st.selectbox(
    "Choose a graph type:",
    ["Box and Whisker", "Violin", "Average ± Error Bars", "Strip (Jitter)"]
)
scale_option = st.radio(
    "Choose y-axis scale:",
    ["Linear", "Logarithmic"],
    horizontal=True,
    key="scale_class_compare"
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


# --- Section: Explore Relationships Between Variables ---
st.header("D) Explore Relationships Between Variables")

st.write("""
Now that you’ve explored single variables and compared groups, it’s time to look for **relationships between two variables**.

In this section, you’ll examine how **brain size** and **body mass** relate across animal classes — and how different scaling choices can change what you see.
""")

# --- Prepare data ---
plot_data = data.copy()
plot_data["Class (common name)"] = plot_data["class"].map({
    "Amphibia": "Amphibians",
    "Arachnida": "Spiders & Scorpions",
    "Aves": "Birds",
    "Insecta": "Insects",
    "Malacostraca": "Crustaceans",
    "Mammalia": "Mammals",
    "Clitellata": "Worms",
    "Gastropoda": "Snails & Slugs",
    "Reptilia": "Reptiles"
})
plot_data = plot_data.dropna(subset=["body mass (kg)", "brain size (kg)", "Class (common name)"])
plot_data = plot_data[(plot_data["body mass (kg)"] > 0) & (plot_data["brain size (kg)"] > 0)]

classes = plot_data["Class (common name)"].unique()
colours = plt.cm.tab10(np.linspace(0, 1, len(classes)))

# --- User controls ---
scale_option = st.radio(
    "Choose axis scale:",
    ["Linear", "Log X", "Log Y", "Log-Log"],
    horizontal=True,
    key="scale_relationship"
)

show_fit = st.checkbox("Show a line of best fit for each class", value=True)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

for cls, color in zip(classes, colours):
    subset = plot_data[plot_data["Class (common name)"] == cls]
    ax.scatter(
        subset["body mass (kg)"],
        subset["brain size (kg)"],
        label=cls,
        alpha=0.7,
        s=25,
        color=color,
        edgecolor="black",
        linewidth=0.2
    )
    if show_fit and len(subset) > 2:
        # Fit line (on log scale if log-log)
        x = subset["body mass (kg)"]
        y = subset["brain size (kg)"]
        if "Log" in scale_option:
            x = np.log10(x)
            y = np.log10(y)
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        if "Log" in scale_option:
            x_line = 10**x_line
            y_line = 10**y_line
        ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.9)

# --- Axis scaling ---
if scale_option in ["Log X", "Log-Log"]:
    ax.set_xscale("log")
if scale_option in ["Log Y", "Log-Log"]:
    ax.set_yscale("log")

# --- Formatting ---
ax.set_xlabel("Body Mass (kg)")
ax.set_ylabel("Brain Size (kg)")
ax.set_title("Brain Size vs Body Mass by Animal Class")
ax.legend(fontsize=8, frameon=True)
ax.grid(True, which="major", linestyle="--", alpha=0.3)
ax.grid(True, which="minor", linestyle=":", alpha=0.1)
fig.tight_layout()

st.pyplot(fig, use_container_width=True)

# --- Reflection prompt ---
st.markdown("""
💬 **Reflection:**  
Each point represents an animal. Lines show how **brain size changes with body mass** within each class.  
When both axes are on a *logarithmic scale*, a straight line means a **power-law relationship** (e.g., “brain size grows in proportion to body mass”).  

✏️ **Questions to think about (and answer in the sidebar):**  
- Which animal groups have the steepest (or flattest) slopes?  
- What might that mean about how brains scale with body size?  
- How do log scales make the relationship easier to interpret?
""")

