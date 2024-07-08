import hashlib
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from typing import List
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, func
from datetime import datetime
import uvicorn

SQLALCHEMY_DATABASE_URL = "//"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UserGet(BaseModel):
    age: int
    city: str
    country: str
    exp_group: int
    gender: int
    id: int
    os: str
    source: str

    class Config:
        orm_mode = True

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class FeedGet(BaseModel):
    action: str
    post_id: int
    time: datetime
    user_id: int
    user: UserGet
    post: PostGet

    class Config:
        orm_mode = True


class Post(Base):
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

class PostsInfo(BaseModel):
    post_id : int
    TextCluster : int
    DistanceToCluster_0 : float
    DistanceToCluster_1 : float
    DistanceToCluster_2 : float
    DistanceToCluster_3 : float
    DistanceToCluster_4 : float
    DistanceToCluster_5 : float
    DistanceToCluster_6 : float
    DistanceToCluster_7 : float
    DistanceToCluster_8 : float
    DistanceToCluster_9 : float
    DistanceToCluster_10 :float
    DistanceToCluster_11 : float
    DistanceToCluster_12 : float
    DistanceToCluster_13 : float
    DistanceToCluster_14 : float

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


SALT = "recommend_salt"
GROUP_A_PERCENTAGE = 50

def get_exp_group(user_id: int) -> str:

    input_data = str(user_id) + SALT

    hashed_data = hashlib.md5(input_data.encode()).hexdigest()

    hash_int = int(hashed_data, 16)

    if hash_int % 100 < GROUP_A_PERCENTAGE:
        return "control"
    else:
        return "test"


def get_model_path(path: str, model_type: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f'/workdir/user_input/{model_type}'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_model_control():
    model_control_path = get_model_path("../models/catboost_model_1", "model_control")
    from_file = CatBoostClassifier()
    model_control = from_file.load_model(model_control_path, format='cbm')

    return model_control


def load_model_test():
    model_test_path = get_model_path("../models/catboost_model_2", "model_test")
    from_file = CatBoostClassifier()
    model_test = from_file.load_model(model_test_path, format='cbm')

    return model_test

model_control = load_model_control()
model_test = load_model_test()


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "//"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:   # загрузка подготовленной таблицы users

    return batch_load_sql('SELECT * FROM evg_user_features_lesson_22')

def load_posts_mod_control() -> pd.DataFrame: # загрузка подготовленной таблицы posts control

    return batch_load_sql('SELECT * FROM evg_post_features_lesson_22')

def load_posts_mod_test() -> pd.DataFrame: # загрузка подготовленной таблицы posts test

    return batch_load_sql('SELECT * FROM public.posts_info_features_dl_efel')

def load_posts() -> pd.DataFrame: # загрузка таблицы all_posts

    return batch_load_sql('SELECT * FROM public.post_text_df')


df_users = load_features()
df_posts = load_posts()
df_posts_mod_control = load_posts_mod_control()
df_posts_mod_test = load_posts_mod_test()

def user_with_all_posts_control(user_id, df_users, df_posts_mod_control): # таблица из одного user и всех posts control

    df_table = df_posts_mod_control.copy()

    for column in df_users.columns:
        df_table[column] = df_users.loc[df_users['user_id'] == user_id, column].values[0]

    df_table = df_table.drop('user_id', axis=1)

    df_table['gender'] = df_table['gender'].astype('int64')
    df_table['age'] = df_table['age'].astype('int64')
    df_table['city'] = df_table['city'].astype('float64')
    df_table['exp_group'] = df_table['exp_group'].astype('int64')
    df_table['os_iOS'] = df_table['os_iOS'].astype('uint8')
    df_table['source_organic'] = df_table['source_organic'].astype('uint8')
    df_table['topic'] = df_table['topic'].astype('float64')
    df_table['time_of_day_morning'] = df_table['time_of_day_morning'].astype('uint8')
    df_table['time_of_day_night'] = df_table['time_of_day_night'].astype('uint8')

    return df_table

features_control = ['post_id', 'gender', 'age', 'city', 'exp_group', 'os_iOS', 'source_organic', 'topic', 'time_of_day_morning', 'time_of_day_night']

def user_with_all_posts_test(user_id, df_users, df_posts_mod_test): # таблица из одного user и всех posts test

    df_table = df_posts_mod_test.copy()

    for column in df_users.columns:
        df_table[column] = df_users.loc[df_users['user_id'] == user_id, column].values[0]

    df_table = df_table.drop('user_id', axis=1)

    df_table['gender'] = df_table['gender'].astype('int64')
    df_table['age'] = df_table['age'].astype('int64')
    df_table['city'] = df_table['city'].astype('float64')
    df_table['exp_group'] = df_table['exp_group'].astype('int64')
    df_table['os_iOS'] = df_table['os_iOS'].astype('int64')
    df_table['source_organic'] = df_table['source_organic'].astype('int64')
    df_table['time_of_day_morning'] = df_table['time_of_day_morning'].astype('int64')
    df_table['time_of_day_night'] = df_table['time_of_day_night'].astype('int64')
    df_table['post_id'] = df_table['post_id'].astype('int64')
    df_table['TextCluster'] = df_table['TextCluster'].astype('int32')
    df_table['DistanceToCluster_0'] = df_table['DistanceToCluster_0'].astype('float32')
    df_table['DistanceToCluster_1'] = df_table['DistanceToCluster_1'].astype('float32')
    df_table['DistanceToCluster_2'] = df_table['DistanceToCluster_2'].astype('float32')
    df_table['DistanceToCluster_3'] = df_table['DistanceToCluster_3'].astype('float32')
    df_table['DistanceToCluster_4'] = df_table['DistanceToCluster_4'].astype('float32')
    df_table['DistanceToCluster_5'] = df_table['DistanceToCluster_5'].astype('float32')
    df_table['DistanceToCluster_6'] = df_table['DistanceToCluster_6'].astype('float32')
    df_table['DistanceToCluster_7'] = df_table['DistanceToCluster_7'].astype('float32')
    df_table['DistanceToCluster_8'] = df_table['DistanceToCluster_8'].astype('float32')
    df_table['DistanceToCluster_9'] = df_table['DistanceToCluster_9'].astype('float32')
    df_table['DistanceToCluster_10'] = df_table['DistanceToCluster_10'].astype('float32')
    df_table['DistanceToCluster_11'] = df_table['DistanceToCluster_11'].astype('float32')
    df_table['DistanceToCluster_12'] = df_table['DistanceToCluster_12'].astype('float32')
    df_table['DistanceToCluster_13'] = df_table['DistanceToCluster_13'].astype('float32')
    df_table['DistanceToCluster_14'] = df_table['DistanceToCluster_14'].astype('float32')
    df_table['topic_covid'] = df_table['topic_covid'].astype('int64')
    df_table['topic_entertainment'] = df_table['topic_entertainment'].astype('int64')
    df_table['topic_movie'] = df_table['topic_movie'].astype('int64')
    df_table['topic_politics'] = df_table['topic_politics'].astype('int64')
    df_table['topic_sport'] = df_table['topic_sport'].astype('int64')
    df_table['topic_tech'] = df_table['topic_tech'].astype('int64')

    return df_table

features_test = ['gender', 'age', 'city', 'exp_group', 'os_iOS', 'source_organic', 'time_of_day_morning', 'time_of_day_night', 'post_id', 'TextCluster',
            'DistanceToCluster_0', 'DistanceToCluster_1', 'DistanceToCluster_2', 'DistanceToCluster_3', 'DistanceToCluster_4', 'DistanceToCluster_5',
            'DistanceToCluster_6', 'DistanceToCluster_7', 'DistanceToCluster_8', 'DistanceToCluster_9', 'DistanceToCluster_10', 'DistanceToCluster_11',
            'DistanceToCluster_12', 'DistanceToCluster_13', 'DistanceToCluster_14', 'topic_covid', 'topic_entertainment', 'topic_movie',
            'topic_politics', 'topic_sport', 'topic_tech']

app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db



@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5) -> Response:
    exp_group = get_exp_group(id)

    if exp_group == 'control':

        # таблица для предсказаний
        df_table = user_with_all_posts_control(id, df_users, df_posts_mod_control)[features_control]

        # массив из 5 id постов
        post_id = pd.concat([df_table['post_id'], pd.DataFrame(model_control.predict_proba(df_table).T[1], columns=['prediction'])], axis=1).sort_values(by=['prediction'], ascending=False).head(5)['post_id'].values

    elif exp_group == 'test':
        # таблица для предсказаний
        df_table = user_with_all_posts_test(id, df_users, df_posts_mod_test)[features_test]

        # массив из 5 id постов
        post_id = pd.concat([df_table['post_id'], pd.DataFrame(model_test.predict_proba(df_table).T[1], columns=['prediction'])], axis=1).sort_values(by=['prediction'], ascending=False).head(5)['post_id'].values

    else:
        raise ValueError('unknown group')

    # нужные посты из таблицы постов
    result_table = df_posts[df_posts['post_id'].isin(post_id)]

    recommended_posts = []
    for i in range(min(limit, len(result_table))):
        recommended_posts.append(PostGet(id=result_table['post_id'].iloc[i],
                              text=result_table['text'].iloc[i],
                              topic=result_table['topic'].iloc[i]))
    if not recommended_posts:
        raise HTTPException(404, "posts not found")
    response = Response(exp_group=exp_group, recommendations=recommended_posts)
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
