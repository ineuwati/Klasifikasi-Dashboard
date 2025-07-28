import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

st.set_page_config(page_title="Dashboard Klasifikasi Opini KIP Kuliah di X", layout="wide")

# --- Load Data ---
df_preprocessing = pd.read_csv('tweet_preprocessing2.csv')
df_prediction = pd.read_csv('model_prediction.csv')
eval_df = pd.read_csv('evaluation_scores.csv')
summary_df = pd.read_csv('summary_data_pipeline.csv')
cm_nb = pd.read_csv('conf_matrix_nb.csv', header=None).values
cm_svm = pd.read_csv('conf_matrix_svm.csv', header=None).values

conf_matrix_dict = {
    "Naive Bayes": {"matrix": cm_nb, "labels": ["negatif", "positif"]},
    "SVM": {"matrix": cm_svm, "labels": ["negatif", "positif"]}
}

st.markdown("<h1 style='text-align: center;'>Klasifikasi Opini Program KIP Kuliah di X (Twitter)</h1>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Summary Data", "🔍 Sentimen Predict"])

# ========================================================================
# =================     TAB 1: SUMMARY DATA    ==========================
# ========================================================================
with tab1:
    st.markdown("#### Jumlah Data Tweet per Bulan (Pra-Modeling)")
    df_preprocessing['created_at'] = pd.to_datetime(df_preprocessing['created_at'])
    df_preprocessing['bulan'] = df_preprocessing['created_at'].dt.to_period("M").astype(str)
    tweets_per_bulan = df_preprocessing.groupby('bulan').size().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tweets_per_bulan.index, tweets_per_bulan.values)
    plt.ylabel("Jumlah Tweet")
    plt.xlabel("Bulan")
    plt.title("Jumlah Tweet per Bulan (Hasil Preprocessing dan label)")
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    st.pyplot(fig)

    summary_df_show = summary_df.copy()
    summary_df_show['Persentase (%)'] = summary_df_show['Persentase (%)'].map('{:.2f}%'.format)

    st.markdown("#### Distribusi Data: Preprocessing, Split, SMOTE")
    st.table(summary_df_show)
    st.caption("Tabel ini menampilkan jumlah data setelah preprocessing, split, dan SMOTE pada pipeline.")

# ========================================================================
# ======================   TAB 2: SENTIMEN PREDICT   ====================
# ========================================================================
with tab2:
    df_preprocessing['id'] = df_preprocessing.index
    reverse_label_map = {0: 'negatif', 1: 'positif'}

    df_long = pd.DataFrame()
    for model in ['Naive Bayes', 'SVM']:
        col = 'pred_nb' if model == 'Naive Bayes' else 'pred_svm'
        temp = df_prediction[['id', 'label', col]].copy()
        temp = temp.rename(columns={col: 'predicted'})
        temp['model'] = model
        df_long = pd.concat([df_long, temp], axis=0)

    df_long['predicted_label'] = df_long['predicted'].map(reverse_label_map)
    df_long['actual_label'] = df_long['label']
    df_long = df_long.drop(columns=['label', 'predicted'])

    df = pd.merge(df_long, df_preprocessing[['id', 'tweet', 'created_at']], on='id', how='left')
    df['created_at'] = pd.to_datetime(df['created_at'])

    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    with fcol1:
        date_range = st.date_input("Periode:", [df['created_at'].min(), df['created_at'].max()])
    with fcol2:
        selected_sentiment = st.multiselect("Sentimen:", df['predicted_label'].unique(), default=list(df['predicted_label'].unique()))
    with fcol3:
        selected_model = st.multiselect("Model:", df['model'].unique(), default=list(df['model'].unique()))

    filtered_df = df[
        (df['model'].isin(selected_model)) &
        (df['predicted_label'].isin(selected_sentiment)) &
        (df['created_at'].dt.tz_localize(None).between(
            pd.to_datetime(date_range[0]),
            pd.to_datetime(date_range[1])
        ))
    ]

    # Tambah kolom: TP, TN, FP, FN
    def get_prediction_type(row):
        pred = row['predicted_label']
        actual = row['actual_label']
        if pred == 'positif' and actual == 'positif':
            return 'TP'
        elif pred == 'negatif' and actual == 'negatif':
            return 'TN'
        elif pred == 'positif' and actual == 'negatif':
            return 'FP'
        elif pred == 'negatif' and actual == 'positif':
            return 'FN'
        else:
            return 'Unknown'

    filtered_df['prediction_type'] = filtered_df.apply(get_prediction_type, axis=1)

    sc1, sc2, sc3 = st.columns(3)
    nb_df = df[
        (df['model'] == 'Naive Bayes') &
        (df['predicted_label'].isin(selected_sentiment)) &
        (df['created_at'].dt.tz_localize(None).between(
            pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        ))
    ]
    svm_df = df[
        (df['model'] == 'SVM') &
        (df['predicted_label'].isin(selected_sentiment)) &
        (df['created_at'].dt.tz_localize(None).between(
            pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        ))
    ]
    acc_nb = (nb_df['predicted_label'] == nb_df['actual_label']).mean() * 100 if not nb_df.empty else 0
    acc_svm = (svm_df['predicted_label'] == svm_df['actual_label']).mean() * 100 if not svm_df.empty else 0
    sc1.metric("Akurasi NB", f"{acc_nb:.2f}%")
    sc2.metric("Akurasi SVM", f"{acc_svm:.2f}%")
    sc3.metric("Total Tweet", len(filtered_df))

    st.write("")
    if filtered_df.empty:
        st.info("Tidak ada data yang sesuai filter.")
    else:
        st.markdown("#### 📊 Perbandingan Metrik Evaluasi (Seluruh Data)")
        st.caption("*Metrik di bawah ini dihitung dari seluruh data test.*")
        colA, colB = st.columns([2, 2])
        with colA:
            score_to_plot = st.selectbox("Pilih Metrik", ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
            fig_barchart, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Model', y=score_to_plot, data=eval_df, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title(f"Perbandingan {score_to_plot}")
            for i, v in enumerate(eval_df[score_to_plot]):
                ax.text(i, v + 0.02, f"{v:.2%}", color='black', fontweight='bold', ha='center')
            st.pyplot(fig_barchart)
        with colB:
            model_conf = st.selectbox("Pilih Model (Confusion Matrix)", list(conf_matrix_dict.keys()), key="conf_mat")
            cm = conf_matrix_dict[model_conf]['matrix']
            labels = conf_matrix_dict[model_conf]['labels']
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
            plt.title(f'Confusion Matrix {model_conf}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt.gcf())

    dcol1, dcol2 = st.columns([1, 2])
    with dcol1:
        st.markdown("#### Distribusi Sentimen")
        if filtered_df.empty:
            st.info("Tidak ada data tersedia")
        else:
            plt.figure(figsize=(4, 4))
            filtered_df['predicted_label'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
            plt.ylabel("")
            st.pyplot(plt.gcf())
    with dcol2:
        st.markdown("#### Jumlah Tweet per Sentimen")
        if filtered_df.empty:
            st.info("Tidak ada data tersedia")
        else:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=filtered_df, x='predicted_label', hue='model')
            plt.title("")
            st.pyplot(plt.gcf())

    # WORD CLOUD
    st.write("")
    wcol1, wcol2 = st.columns(2)
    for sentiment, col in zip(['positif', 'negatif'], [wcol1, wcol2]):
        with col:
            st.markdown(f"#### Word Cloud ({sentiment.capitalize()})")
            wc_df = filtered_df[filtered_df['predicted_label'] == sentiment]
            if not wc_df.empty:
                text = " ".join(wc_df['tweet'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(5, 3))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())
            else:
                st.write("Tidak ada data")

    # DETAIL DATA TABLE
    st.write("")
    st.markdown("#### Table: Detail Tweet, Sentimen, Model, dan Tipe Prediksi")
    st.dataframe(
        filtered_df[['created_at', 'tweet', 'predicted_label', 'actual_label', 'model', 'prediction_type']],
        use_container_width=True
    )
