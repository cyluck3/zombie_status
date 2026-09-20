import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


@st.cache_data
def generate_synthetic_data(size=1500, random_seed=42):
    np.random.seed(random_seed)
    
    dataset = {
        "gender": np.random.choice(["male", "female"], size=size),
        "age": np.random.randint(18, 80, size=size),
        "exposed_to_infected_ones": np.random.randint(1, 100, size=size),
        "temperature": np.random.randint(35, 50, size=size) * np.random.normal(1, 0.09, size=size),
        "bites": np.random.randint(0, 5, size=size),
        "speaking_coherence": np.random.randint(1, 10, size=size),
        "movements_coherence": np.random.randint(1, 10, size=size),
        "impulsiveness": np.random.randint(1, 10, size=size),
        "red_eyes": np.random.choice(["yes", "no"], size=size),
        "hunger": np.random.randint(1, 10, size=size)
    }

    df = pd.DataFrame(dataset)
    df["hours_after_bitten"] = np.where(df.bites != 0, np.random.randint(1, 24, size=size), 0)

    score = (
        1.0 * (df["gender"] == "male").astype(int)
        + 0.05 * (df["age"] - 40)
        + 0.08 * (df["exposed_to_infected_ones"] - 50)
        + 0.3 * (df["temperature"] - 37.0)
        + 1.5 * (df["bites"])
        - 0.8 * (df["speaking_coherence"] - 5)
        - 0.8 * (df["movements_coherence"] - 5)
        + 0.5 * (df["impulsiveness"] - 5)
        + 1.0 * (df["red_eyes"] == "yes").astype(int)
        + 0.5 * (df["hunger"] - 5)
        + 0.2 * (df["hours_after_bitten"])
    )

    score_scaled = (score - score.mean()) / score.std()
    probability = 1 / (1 + np.exp(-score_scaled))
    df["about_to_transform"] = np.random.binomial(1, probability)

    return df


@st.cache_resource
def train_models(df):
    X = df.drop("about_to_transform", axis=1)
    y = df["about_to_transform"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_features = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_features = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_features, [
            "age", "exposed_to_infected_ones", "temperature", "bites", 
            "speaking_coherence", "movements_coherence", "impulsiveness", 
            "hunger", "hours_after_bitten"
        ]),
        ("cat", cat_features, ["gender", "red_eyes"])
    ])

    lr_model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)), 
        ("classifier", LogisticRegression(solver='liblinear')) 
    ])

    rf_model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)), 
        ("classifier", RandomForestClassifier(random_state=42)) 
    ])

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    eval_results = {
        "lr_accuracy": accuracy_score(y_test, y_pred_lr),
        "lr_report": classification_report(y_test, y_pred_lr, output_dict=True),
        "rf_accuracy": accuracy_score(y_test, y_pred_rf),
        "rf_report": classification_report(y_test, y_pred_rf, output_dict=True)
    }

    return lr_model, rf_model, eval_results


class ZombieStatusApp:
    def __init__(self):
        self.config()
        self.df = generate_synthetic_data()
        self.lr_model, self.rf_model, self.eval_results = train_models(self.df)

    def config(self):
        st.set_page_config(page_title="Zombie Status", layout="wide")

    def run(self):
        self.sidebar_inputs()
        self.header()
        self.body()
        self.footer()

    def sidebar_inputs(self):
        with st.sidebar:
            st.title("⚙️ Settings")
            st.write("---")

            self.age = st.slider("Age", 1, 120, 40)
            self.gender = st.selectbox("Gender", ["male", "female"])
            self.hunger = st.slider("Hunger", 1, 10, 6)
            self.impulsiveness = st.slider("Impulsiveness", 0, 10, 4)
            self.red_eyes = st.selectbox("Red Eyes", ["yes", "no"])
            self.temperature = st.slider("Temperature", 1, 50, 35)
            self.exposed_to_infected_ones = st.slider("Exposed to infected ones", 6, 100, 41)
            self.bites = st.slider("Bites", 0, 10, 3)
            self.hours = st.slider("Hours after bitten", 0, 24, 4)
            self.speaking_coherence = st.slider("Speaking Coherence", 1, 10, 8)
            self.movements_coherence = st.slider("Movement Coherence", 1, 10, 9)

    def get_user_dataframe(self):
        return pd.DataFrame({
            "gender": [self.gender],
            "age": [self.age],
            "exposed_to_infected_ones": [self.exposed_to_infected_ones],
            "temperature": [self.temperature],
            "bites": [self.bites],
            "speaking_coherence": [self.speaking_coherence],
            "movements_coherence": [self.movements_coherence],
            "impulsiveness": [self.impulsiveness],
            "red_eyes": [self.red_eyes],
            "hunger": [self.hunger],
            "hours_after_bitten": [self.hours if self.bites > 0 else 0]
        })

    def header(self):
        st.title("Becoming a Zombie? 🧟‍♂️ - Logistic Regression Model")
        st.markdown("*These models predict if the person is about to transform into a zombie or not.*")

        self.logistic_model_()
        st.write("---")
        self.random_forest_classifier()
        st.write("---")
        st.header("As for the Dataset (Synthetic Data)")
        self.metrics()

    def logistic_model_(self):
        user_df = self.get_user_dataframe()
        result = self.lr_model.predict(user_df)[0]
        prob = float(self.lr_model.predict_proba(user_df)[0][1])

        st.header("Predictions")
        co1, co2 = st.columns(2)

        with co1:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Prediction", result, f"Score: {prob * 100:.1f}", delta_color="red" if result == 1 else "normal")
                st.progress(prob, text="Points for transforming soon")
            with sc2:
                st.metric("You are a" if result == 1 else "Not a", "Zombie", "+" if result == 1 else "-", delta_color="red" if result == 1 else "normal")

        with co2:
            st.metric("Accuracy with Smote", self.eval_results["lr_accuracy"], delta_description="LG Model")
            st.progress(self.eval_results["lr_accuracy"], text="Accuracy")

        st.dataframe(self.eval_results["lr_report"])

    def random_forest_classifier(self):
        st.title("Let's try a Random Forest Classifier")

        user_df = self.get_user_dataframe()
        result = self.rf_model.predict(user_df)[0]
        prob = float(self.rf_model.predict_proba(user_df)[0][1])

        st.header("Predictions")
        co1, co2 = st.columns(2)

        with co1:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Prediction", result, f"Score: {prob * 100:.1f}", delta_color="red" if result == 1 else "normal")
                st.progress(prob, text="Points for transforming soon")
            with sc2:
                st.metric("You are a" if result == 1 else "Not a", "Zombie", "+" if result == 1 else "-", delta_color="red" if result == 1 else "normal")

        with co2:
            st.metric("Accuracy with Smote", self.eval_results["rf_accuracy"], delta_description="RF Model")
            st.progress(self.eval_results["rf_accuracy"], text="Accuracy")

        st.dataframe(self.eval_results["rf_report"])

    def metrics(self):
        co1, co2, co3, co4 = st.columns(4)

        with co1:
            st.metric("Total Samples", self.df.shape[0], delta_description="People")
        with co2:
            males = (self.df["gender"] == "male").sum()
            st.metric("Male Samples", males, f"{males / len(self.df):.2%}")
        with co3:
            females = (self.df["gender"] == "female").sum()
            st.metric("Female Samples", females, f"{females / len(self.df):.2%}")
        with co4:
            st.metric("Mean Age", int(self.df.age.mean()), delta_description="Years old")

        st.write("---")

        co1, co2, co3, co4 = st.columns(4)

        not_infected_count = (self.df["about_to_transform"] == 0).sum()
        infected_count = (self.df["about_to_transform"] == 1).sum()

        with co1:
            n_m_not = ((self.df["gender"] == "male") & (self.df["about_to_transform"] == 0)).sum()
            st.metric("M - Not becoming yet", n_m_not, f"{n_m_not / not_infected_count:.2%}")

        with co2:
            n_f_not = ((self.df["gender"] == "female") & (self.df["about_to_transform"] == 0)).sum()
            st.metric("F - Not becoming yet", n_f_not, f"{n_f_not / not_infected_count:.2%}")

        with co3:
            n_m_inf = ((self.df["gender"] == "male") & (self.df["about_to_transform"] == 1)).sum()
            st.metric("M - About to become", n_m_inf, f"{n_m_inf / infected_count:.2%}", delta_color="inverse")

        with co4:
            n_f_inf = ((self.df["gender"] == "female") & (self.df["about_to_transform"] == 1)).sum()
            st.metric("F - About to become", n_f_inf, f"{n_f_inf / infected_count:.2%}", delta_color="inverse")

    def body(self):
        self.graphics()

    def graphics(self):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bites", "Impulsiveness", "Hunger", "Red Eyes", "Temperature"])
        
        with tab1:
            fig = px.histogram(self.df, x="age", y="bites", color="gender", color_discrete_map={"male": "#2FB2D6", "female": "#79D0F2"})
            fig.update_layout(title="Number of Bites by Age and Gender", xaxis_title="Age", yaxis_title="Number of Bites", legend_title="Gender")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.histogram(self.df, x="age", y="impulsiveness", color="gender", color_discrete_map={"male": "#D62F2F", "female": "#FF7A7A"})
            fig.update_layout(title="Level of Impulsiveness by Age and Gender", xaxis_title="Age", yaxis_title="Level of Impulsiveness", legend_title="Gender")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = px.histogram(self.df, x="age", y="hunger", color="gender", color_discrete_map={"male": "#F54927", "female": "#FF8E7A"})
            fig.update_layout(title="Level of Hunger by Age and Gender", xaxis_title="Age", yaxis_title="Level of Hunger", legend_title="Gender")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = px.histogram(self.df, x="red_eyes", y="age", color="gender", color_discrete_map={"male": "#BFB01B", "female": "#F0E262"})
            fig.update_layout(title="Red Eyes by Age and Gender", xaxis_title="Red Eyes", yaxis_title="Age", legend_title="Gender")
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            df_temp = self.df.groupby("gender")["temperature"].count().reset_index()
            fig = px.histogram(df_temp, x="gender", y="temperature", color="gender", color_discrete_map={"male": "#F55427", "female": "#FFA67A"})
            fig.update_layout(title="Level of body temperature by Gender", xaxis_title="Gender", yaxis_title="Level of temperature", legend_title="Gender")
            st.plotly_chart(fig, use_container_width=True)

    def footer(self):
        st.header("Raw Data for Training")
        st.dataframe(self.df)


if __name__ == "__main__":
    app = ZombieStatusApp()
    app.run()