import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import spatial
from pyspark.sql.functions import lower, col


def split_genres(rdd):
    return rdd.map(lambda x: (x.movieId, x.genres)) \
        .keyBy(lambda x: x[0]) \
        .mapValues(lambda x: x[1].lower().split('|')) \
        .flatMapValues(lambda x: x)


def count_genres(rdd):
    return rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)


def ratings_stats(rdd_genres, rdd_ratings):
    return rdd_genres.join(rdd_ratings.map(lambda x: (x.movieId, x.rating))) \
        .map(lambda x: (x[1][0], (x[1][1], 1))) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .map(lambda x: (x[0], *x[1], x[1][0] / x[1][1]))


def contains_element(x, titles):
    for title in titles.value:
        if title[0] in x:
            return title[1]
    return None


def get_films_ids(df, titles):
    return df.withColumn('title_lower', lower(col('title'))) \
        .rdd.map(lambda x: (x.movieId, x.title_lower)) \
        .map(lambda x: (x[0], contains_element(x[1], titles))) \
        .filter(lambda x: x[1])


def ratings_similarity_MSE(l, d, v):
    result = np.zeros(len(d.value))
    for movieId, rating in l:
        result[d.value[movieId]] = rating
    return mean_squared_error(result, v.value)


def ratings_similarity_COS(l, d, v):
    result = np.zeros(len(d.value))
    for movieId, rating in l:
        result[d.value[movieId]] = rating
    sim = 1 - spatial.distance.cosine(result, v.value)
    return 2 / (1 / sim + 1 / len(l))


def get_similarity(rdd, db, vb, ids, top=10, similarity_function=ratings_similarity_MSE):
    return rdd.filter(lambda x: x.movieId in ids.value) \
        .map(lambda x: (x.userId, [(x.movieId, x.rating)])) \
        .reduceByKey(lambda a, b: a + b) \
        .mapValues(lambda x: similarity_function(x, db, vb)) \
        .takeOrdered(top, key=lambda x: x[1])


def harmonic(a, b):
    return 2 / (1 / a + 1 / b)


def get_films_stats(rdd_users, rdd_ratings, seen_films_id, top=10):
    tmp = rdd_ratings.filter(lambda x: x.movieId not in seen_films_id.value) \
        .keyBy(lambda x: x.userId) \
        .mapValues(lambda x: (x.movieId, x.rating))
    return rdd_users.join(tmp) \
        .map(lambda x: x[1][1]) \
        .mapValues(lambda x: (x, 1)) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda x: (x[0] / x[1], x[1])) \
        .mapValues(lambda x: (*x, harmonic(x[0], x[1]))) \
        .top(top, key=lambda x: x[1][2])


def get_recommendations(rdd_films_stats, rdd_films):
    columns = ['rating', 'users_seen', 'f_score', 'title', 'genres']
    rdd_films_data = rdd_films.map(lambda x: (x.movieId, (x.title, x.genres)))
    return rdd_films_stats.join(rdd_films_data) \
        .map(lambda x: (*x[1][0], *x[1][1])) \
        .toDF(columns).orderBy('f_score', ascending=False)


def recommendations_pipe(sc, df_movies, df_ratings, titles, n_top_users=10,
                         n_top_films=10, similarity_function=ratings_similarity_MSE):
    titles_b = sc.broadcast(titles)
    ids_and_ratings = get_films_ids(df_movies, titles_b).collect()
    titles_b.unpersist()
    ids, ratings = zip(*ids_and_ratings)
    positions_b = sc.broadcast({k: v for v, k in enumerate(ids)})
    ratings_b = sc.broadcast(ratings)
    seen_films_id_b = sc.broadcast(ids)
    top_user = get_similarity(df_ratings.rdd, positions_b, ratings_b, seen_films_id_b,
                              n_top_users, similarity_function)
    ratings_b.unpersist()
    positions_b.unpersist()
    rdd_users = sc.parallelize(top_user)
    films_stats = get_films_stats(rdd_users, df_ratings.rdd, seen_films_id_b, n_top_films)
    seen_films_id_b.unpersist()
    rdd_films_stats = sc.parallelize(films_stats)
    return get_recommendations(rdd_films_stats, df_movies.rdd)
