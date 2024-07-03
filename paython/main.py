import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# تحميل بيانات الأفلام والتقييمات
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# بناء نموذج التوصيات بناءً على التصفية التعاونية
def collaborative_filtering(movie_user_matrix, movie_id):
    cosine_sim = cosine_similarity(movie_user_matrix, movie_user_matrix)
    sim_scores = list(enumerate(cosine_sim[movie_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # أفضل 10 أفلام مشابهة
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices 

# إنشاء مصفوفة المستخدم-الفيلم
movie_user_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

# واجهة المستخدم باستخدام Streamlit
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .stButton>button {
        background-color: rgb(230, 102, 207);
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 10px 5px;
        cursor: pointer;
        border-radius: 10px;
    }
    .stSelectbox>div>div>div {
        font-size: 20px;
        background-color: rgb(226, 189, 219);
        color: #212121;
        border: 2px solid rgb(226, 189, 219);
        border-radius: 10px;
    }
    .st.write {
        background-color: #ffffff;
        border: 4px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
    }
    .table-container {
        margin-top: 20px;
        padding: 10px;
        border-radius: 8px;
        background-color: #ffffff;
        border: 2px solid #ddd;
    }
    .sidebar {
        position: fixed;
        top: 0;
        left: 0;
        width: 200px;
        height: 100%;
        background-color: #333;
        padding: 20px;
        color: white;
    }
    .sidebar a {
        text-decoration: none;
        color: white;
        display: block;
        padding: 10px;
        margin: 10px 0;
    }
    .sidebar a:hover {
        background-color: #444;
    }
    .content {
        margin-left: 220px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# صورة الرأس
st.image("download (1).jpeg", use_column_width=True)

# العنوان الرئيسي
st.title("Free Palestine 🇵🇸")
st.title("🎬  توصيات الأفلام 🎥")

# اختيار الفيلم
selected_movie = st.selectbox("اختر فيلمًا تحبه:", movies['title'].values)

# زر التقديم والعرض
if st.button("تقديم الاقتراحات"):
    movie_id = movies[movies['title'] == selected_movie].index[0]
    recommended_movies = collaborative_filtering(movie_user_matrix, movie_id)
    
    st.write("إليك بعض الأفلام التي قد تعجبك:")
    
    recommended_movies_df = movies.iloc[recommended_movies][['title']]
    recommended_movies_df = recommended_movies_df.reset_index(drop=True)
    
    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
    st.table(recommended_movies_df)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
