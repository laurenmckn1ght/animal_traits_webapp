
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
st.caption(f"Displaying contents of **{DEFAULT_CSV}**.")
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

# Dropdown for variable selection
variable_option = st.selectbox(
    "Choose a variable to plot:",
    options=["body mass (kg)", "brain size (kg)"],
    index=0
)

# Input for number of bins
buckets_option = st.number_input(
    "Enter the number of buckets (bins) to divide your data into:",
    min_value=5,
    max_value=100,
    value=20
)

# Radio button for scale selection
scale_option = st.radio(
    "Choose the scale for the x-axis:",
    options=["Linear", "Logarithmic"],
    horizontal=True
)

# Labels for plotting
labels_list = {
    'body mass (kg)': 'Body Mass (kilograms)',
    'brain size (kg)': 'Brain Size (kilograms)'
}
label = labels_list[variable_option]

# Prepare data (remove NaN and non-positive for log scale)
plot_data = data[variable_option].dropna()
if scale_option == "Logarithmic":
    plot_data = plot_data[plot_data > 0]

# Create histogram
fig, ax = plt.subplots(figsize=(6, 4))  # width, height in inches
ax.hist(plot_data, bins=int(buckets_option), color="#1f77b4", edgecolor="black")
ax.set_xlabel(label)
ax.set_ylabel("Frequency")
ax.set_title(f"Distribution of {label} ({scale_option} scale)", fontsize=12)
ax.tick_params(axis="both", labelsize=10)

# Apply log scaling if chosen
if scale_option == "Logarithmic":
    ax.set_xscale("log")

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
Different types of graphs help us see how each group’s data are distributed and whether some groups tend to be heavier or lighter than others.
""")

# --- Map scientific class names to familiar common names ---
class_labels = {
    "Amphibia": "Amphibians",
    "Arachnida": "Spiders & Scorpions",
    "Aves": "Birds",
    "Insecta": "Insects",
    "Malacostraca": "Crustaceans",
    "Mammalia": "Mammals",
    "Clitellata": "Worms",
    "Gastropoda": "Snails & Slugs",
    "Reptilia": "Reptiles",
    "Chilopoda": "Centipedes"
}

# Add a new column with readable names
data["Class (common name)"] = data["class"].map(class_labels).fillna(data["class"])

# --- Choose graph type ---
graph_type = st.selectbox(
    "Choose a graph type to compare the classes:",
    ["Box and Whisker", "Violin Plot", "Swarm Plot", "Overlapping Histograms"]
)

# --- Choose scale ---
scale_option = st.radio(
    "Choose the scale for the x-axis:",
    options=["Linear", "Logarithmic"],
    horizontal=True
)

# --- Prepare data ---
plot_data = data.dropna(subset=["body mass (kg)", "class"]).copy()
if scale_option == "Logarithmic":
    plot_data = plot_data[plot_data["body mass (kg)"] > 0]

# --- Plot depending on graph type ---
fig, ax = plt.subplots(figsize=(7, 4))

if graph_type == "Box and Whisker":
    import seaborn as sns
    sns.boxplot(
        data=plot_data,
        x="body mass (kg)",
        y="Class (common name)",
        orient="h",
        ax=ax,
        palette="Set3",
        fliersize=2,
    )
    ax.set_title("Body Mass by Animal Class (Box & Whisker)")

elif graph_type == "Violin Plot":
    import seaborn as sns
    sns.violinplot(
        data=plot_data,
        x="body mass (kg)",
        y="Class (common name)",
        orient="h",
        ax=ax,
        palette="Set2",
        inner="quartile",
    )
    ax.set_title("Body Mass by Animal Class (Violin Plot)")

elif graph_type == "Swarm Plot":
    import seaborn as sns
    sns.swarmplot(
        data=plot_data,
        x="body mass (kg)",
        y="Class (common name)",
        orient="h",
        ax=ax,
        size=3,
        palette="husl",
    )
    ax.set_title("Body Mass by Animal Class (Swarm Plot)")

else:  # Overlapping histograms
    for cls, group in plot_data.groupby("Class (common name)"):
        ax.hist(
            group["body mass (kg)"],
            bins=20,
            alpha=0.4,
            label=cls,
        )
    ax.legend(fontsize=7)
    ax.set_title("Body Mass by Animal Class (Overlapping Histograms)")

# Apply scale
if scale_option == "Logarithmic":
    ax.set_xscale("log")

ax.set_xlabel("Body Mass (kg)")
ax.set_ylabel("Animal Class")
st.pyplot(fig, use_container_width=True)

# --- Reflection question ---
st.markdown("""
🚦 Try different graph types. Which do you like the best? Answer question 5 in the sidebar.
""")



"""
st.header("B) Visualise relationships between variables")
st.caption("Choose two numeric variables. Try different axis scales and look for linear, curved, or clustered patterns.")
if len(num_cols) < 2:
    st.info("Need at least two numeric columns to plot a relationship.")
else:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        xcol = st.selectbox("X variable", options=num_cols, index=0)
    with c2:
        ycol = st.selectbox("Y variable", options=[c for c in num_cols if c != xcol], index=min(1, len(num_cols)-1))
    with c3:
        color_col = st.selectbox("Colour by (optional)", options=[None] + cat_cols + [c for c in num_cols if c not in [xcol, ycol]], index=0)

    scale = st.radio("Axis scaling", options=["Linear", "Log X", "Log Y", "Log-Log"], horizontal=True)

    plot_df = work_df[[xcol, ycol]].dropna().copy()
    if scale in ["Log X", "Log-Log"]:
        plot_df = plot_df[plot_df[xcol] > 0]
    if scale in ["Log Y", "Log-Log"]:
        plot_df = plot_df[plot_df[ycol] > 0]

    fig = px.scatter(work_df if color_col else plot_df, x=xcol, y=ycol, color=color_col if color_col else None, opacity=0.8)
    if scale in ["Log X", "Log-Log"]:
        fig.update_xaxes(type="log")
    if scale in ["Log Y", "Log-Log"]:
        fig.update_yaxes(type="log")
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("> Write an observation about the shape or pattern you see.")
    obs1 = st.text_area("Observation", placeholder="Describe patterns: linear, curved, clusters, outliers, proportional changes, etc.", key="obs1")

st.subheader("3) Fit a simple linear model and check the fit")
st.caption("This fits y = a + b·x on the chosen variables. If you chose log scales, the model still fits the raw values shown above.")
if len(num_cols) >= 2:
    # Prepare data for linear regression on raw values
    df_fit = work_df[[xcol, ycol]].dropna().copy()
    if not df_fit.empty:
        X = df_fit[[xcol]].values.reshape(-1, 1)
        y = df_fit[ycol].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        a = float(model.intercept_[0])
        b = float(model.coef_[0][0])
        y_pred = model.predict(X).ravel()
        # R^2
        r2 = float(model.score(X, y))

        st.write(f"Fitted model: **{ycol} = {a:.3g} + {b:.3g} × {xcol}**")
        st.write(f"R² = {r2:.3f}")

        # Overlay line
        xs = np.linspace(df_fit[xcol].min(), df_fit[xcol].max(), 100)
        ys = a + b * xs
        line = go.Scatter(x=xs, y=ys, mode="lines", name="Linear fit")
        fig2 = px.scatter(df_fit, x=xcol, y=ycol, opacity=0.75)
        fig2.add_trace(line)
        fig2.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("> Does a straight line describe the relationship well or poorly?")
        obs2 = st.text_area("Comment on the fit", placeholder="Consider R², residual spread, outliers, curvature, or distinct groups.", key="obs2")

        st.subheader("4) Make predictions with your fitted model")
        st.caption("Enter one or more X values to predict Y using the model above.")
        pred_x = st.text_input(f"Values of {xcol} (comma separated)", value="")
        if pred_x.strip():
            try:
                xs_in = np.array([float(v) for v in pred_x.split(",")])
                ys_out = a + b * xs_in
                pred_df = pd.DataFrame({xcol: xs_in, f"pred_{ycol}": ys_out})
                st.dataframe(pred_df, use_container_width=True)
                st.download_button("Download predictions as CSV", pred_df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Could not parse input values: {e}")
    else:
        st.info("Not enough data to fit a model.")

st.subheader("5) Save your observations")
st.caption("Download your notes as a CSV to submit or keep.")
student_name = st.text_input("Your name or initials", "")
notes = {
    "name": student_name,
    "x_variable": xcol if len(num_cols) >= 2 else "",
    "y_variable": ycol if len(num_cols) >= 2 else "",
    "obs_plot": st.session_state.get("obs1", ""),
    "obs_fit": st.session_state.get("obs2", ""),
}
notes_df = pd.DataFrame([notes])
st.download_button("Download my observations (CSV)", notes_df.to_csv(index=False).encode("utf-8"), file_name="observations_notes.csv", mime="text/csv")

st.caption("Tip for teachers: ship an 'observations.csv' with the class dataset, or instruct students to upload one. The app hides all code and focuses on exploration, simple modelling, and note taking.")    
"""