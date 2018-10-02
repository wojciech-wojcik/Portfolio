import findspark

findspark.init()

import logging
import pytest
from pyspark.sql import SparkSession, Row
import spark_functions
import pandas as pd


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_context(request):
    spark = SparkSession.builder.appName("pytest-recommendations") \
        .master("local[2]").getOrCreate()
    sc = spark.sparkContext
    request.addfinalizer(sc.stop)
    quiet_py4j()
    return sc


@pytest.mark.usefixtures('spark_context')
def test_split_genres(spark_context):
    test_input = [Row(movieId=1, title='Toy Story (1995)', genres='Adventure|Animation|Children|Comedy|Fantasy'),
                  Row(movieId=2, title='Jumanji (1995)', genres='Adventure|Children|Fantasy'),
                  Row(movieId=3, title='Grumpier Old Men (1995)', genres='Comedy|Romance'),
                  Row(movieId=4, title='Waiting to Exhale (1995)', genres='Comedy|Drama|Romance'),
                  Row(movieId=5, title='Father of the Bride Part II (1995)', genres='Comedy')]

    input_rdd = spark_context.parallelize(test_input, 2)
    print(input_rdd.collect())
    results = spark_functions.split_genres(input_rdd).collect()
    expected_results = {(1, 'adventure'), (1, 'animation'), (1, 'children'), (1, 'comedy'), (1, 'fantasy'),
                        (2, 'adventure'), (2, 'children'), (2, 'fantasy'),
                        (3, 'comedy'), (3, 'romance'),
                        (4, 'comedy'), (4, 'romance'), (4, 'drama'),
                        (5, 'comedy')}
    assert set(results) == expected_results


@pytest.mark.usefixtures('spark_context')
def test_count_genres(spark_context):
    test_input = [(1, 'adventure'), (1, 'animation'), (1, 'children'), (1, 'comedy'), (1, 'fantasy'),
                  (2, 'adventure'), (2, 'children'), (2, 'fantasy'),
                  (3, 'comedy'), (3, 'romance'),
                  (4, 'comedy'), (4, 'romance'), (4, 'drama'),
                  (5, 'comedy')]

    input_rdd = spark_context.parallelize(test_input, 2)
    results = spark_functions.count_genres(input_rdd).collect()
    expected_results = {'adventure': 2, 'animation': 1, 'children': 2,
                        'comedy': 4, 'fantasy': 2, 'romance': 2, 'drama': 1}
    assert dict(results) == expected_results


@pytest.mark.usefixtures('spark_context')
def test_ratings_stats(spark_context):
    test_input1 = [(1, 'adventure'), (1, 'animation'), (1, 'children'), (1, 'comedy'), (1, 'fantasy'),
                   (2, 'adventure'), (2, 'children'), (2, 'fantasy'),
                   (3, 'comedy'), (3, 'romance'),
                   (4, 'comedy'), (4, 'romance'), (4, 'drama'),
                   (5, 'comedy')]

    test_input2 = [Row(userId=1, movieId=1, rating=1, timestamp=1112486027),
                   Row(userId=1, movieId=2, rating=2, timestamp=1112484676),
                   Row(userId=1, movieId=3, rating=3, timestamp=1112484819),
                   Row(userId=1, movieId=4, rating=4, timestamp=1112484727),
                   Row(userId=1, movieId=5, rating=5, timestamp=1112484580)]

    input_rdd1 = spark_context.parallelize(test_input1, 2)
    input_rdd2 = spark_context.parallelize(test_input2, 2)
    results = spark_functions.ratings_stats(input_rdd1, input_rdd2).collect()
    expected_results = {('animation', 1, 1, 1.0),
                        ('comedy', 13, 4, 3.25),
                        ('children', 3, 2, 1.5),
                        ('fantasy', 3, 2, 1.5),
                        ('romance', 7, 2, 3.5),
                        ('adventure', 3, 2, 1.5),
                        ('drama', 4, 1, 4.0)}
    assert set(results) == expected_results


@pytest.mark.usefixtures('spark_context')
def test_get_films_ids(spark_context):
    test_input = [Row(movieId=1, title='Toy Story (1995)', genres='Adventure|Animation|Children|Comedy|Fantasy'),
                  Row(movieId=2, title='Jumanji (1995)', genres='Adventure|Children|Fantasy'),
                  Row(movieId=3, title='Grumpier Old Men (1995)', genres='Comedy|Romance'),
                  Row(movieId=4, title='Waiting to Exhale (1995)', genres='Comedy|Drama|Romance'),
                  Row(movieId=5, title='Father of the Bride Part II (1995)', genres='Comedy')]

    input_df = spark_context.parallelize(test_input, 2).toDF()
    titles = spark_context.broadcast([('toy story', 5), ('jumanji', 5)])
    results = spark_functions.get_films_ids(input_df, titles).collect()
    titles.unpersist()
    expected_results = [(1, 5), (2, 5)]
    assert set(results) == set(expected_results)


@pytest.mark.usefixtures('spark_context')
def test_get_similarity_no_diff(spark_context):
    test_input = [Row(userId=1, movieId=0, rating=5.0, timestamp=1094785698),
                  Row(userId=1, movieId=1, rating=5.0, timestamp=1011209096),
                  Row(userId=1, movieId=2, rating=5.0, timestamp=994020680),
                  Row(userId=1, movieId=3, rating=5.0, timestamp=1230857185),
                  Row(userId=1, movieId=4, rating=5.0, timestamp=1230788346)]

    input_rdd = spark_context.parallelize(test_input, 2)
    d = dict([(i, i) for i in range(5)])
    db = spark_context.broadcast(d)
    vb = spark_context.broadcast([5] * len(d))
    idsb = spark_context.broadcast([0, 1, 2, 3, 4])
    results = spark_functions.get_similarity(input_rdd, db, vb, idsb)
    idsb.unpersist()
    db.unpersist()
    vb.unpersist()
    expected_results = [(1, 0)]
    assert results == expected_results


@pytest.mark.usefixtures('spark_context')
def test_get_similarity_small_diff(spark_context):
    test_input = [Row(userId=1, movieId=0, rating=5.0, timestamp=1094785698),
                  Row(userId=1, movieId=1, rating=5.0, timestamp=1011209096),
                  Row(userId=1, movieId=2, rating=5.0, timestamp=994020680),
                  Row(userId=2, movieId=3, rating=5.0, timestamp=1230857185),
                  Row(userId=2, movieId=4, rating=5.0, timestamp=1230788346)]

    input_rdd = spark_context.parallelize(test_input, 2)
    d = dict([(i, i) for i in range(5)])
    db = spark_context.broadcast(d)
    vb = spark_context.broadcast([5] * len(d))
    idsb = spark_context.broadcast([0, 1, 2, 3, 4])
    results = spark_functions.get_similarity(input_rdd, db, vb, idsb)
    idsb.unpersist()
    db.unpersist()
    vb.unpersist()
    expected_results = [(1, 10), (2, 15)]
    assert results == expected_results


@pytest.mark.usefixtures('spark_context')
def test_get_similarity_max_diff(spark_context):
    test_input = [Row(userId=1, movieId=0, rating=0, timestamp=1094785698),
                  Row(userId=1, movieId=1, rating=0, timestamp=1011209096),
                  Row(userId=1, movieId=2, rating=0, timestamp=994020680),
                  Row(userId=1, movieId=3, rating=0, timestamp=1230857185),
                  Row(userId=1, movieId=4, rating=0, timestamp=1230788346)]

    input_rdd = spark_context.parallelize(test_input, 2)
    d = dict([(i, i) for i in range(5)])
    db = spark_context.broadcast(d)
    vb = spark_context.broadcast([5] * len(d))
    idsb = spark_context.broadcast([0, 1, 2])
    results = spark_functions.get_similarity(input_rdd, db, vb, idsb)
    idsb.unpersist()
    db.unpersist()
    vb.unpersist()
    expected_results = [(1, 25)]
    assert results == expected_results


@pytest.mark.usefixtures('spark_context')
def test_get_films_stats(spark_context):
    test_input1 = [(1, 0), (2, 0), (3, 1)]

    test_input2 = [Row(userId=1, movieId=0, rating=5, timestamp=1094785698),
                   Row(userId=1, movieId=1, rating=5, timestamp=1011209096),
                   Row(userId=1, movieId=2, rating=5, timestamp=994020680),
                   Row(userId=2, movieId=0, rating=5, timestamp=1230857185),
                   Row(userId=2, movieId=1, rating=5, timestamp=1230788346),
                   Row(userId=2, movieId=3, rating=5, timestamp=1230788346)]

    input_rdd1 = spark_context.parallelize(test_input1, 2)
    input_rdd2 = spark_context.parallelize(test_input2, 2)
    ids = spark_context.broadcast([0, 2])
    results = spark_functions.get_films_stats(input_rdd1, input_rdd2, ids)
    ids.unpersist()
    expected_results = [(1, (5.0, 2, 2.857142857142857)),
                        (3, (5.0, 1, 1.6666666666666667))]
    assert results == expected_results


@pytest.mark.usefixtures('spark_context')
def test_get_recommendations(spark_context):
    test_input1 = [(2, (10, 4.7, 6.394557823129252)),
                   (3, (10, 4.65, 6.348122866894197)),
                   (4, (10, 4.5, 6.206896551724139))]

    test_input2 = [Row(movieId=1, title='Toy Story (1995)', genres='Adventure|Animation|Children|Comedy|Fantasy'),
                   Row(movieId=2, title='Jumanji (1995)', genres='Adventure|Children|Fantasy'),
                   Row(movieId=3, title='Grumpier Old Men (1995)', genres='Comedy|Romance'),
                   Row(movieId=4, title='Waiting to Exhale (1995)', genres='Comedy|Drama|Romance'),
                   Row(movieId=5, title='Father of the Bride Part II (1995)', genres='Comedy')]

    input_rdd1 = spark_context.parallelize(test_input1, 2)
    input_rdd2 = spark_context.parallelize(test_input2, 2)
    results = spark_functions.get_recommendations(input_rdd1, input_rdd2).toPandas()

    data = [[10, 4.7, 6.394557823129252, 'Jumanji (1995)',
             'Adventure|Children|Fantasy'],
            [10, 4.65, 6.348122866894197, 'Grumpier Old Men (1995)',
             'Comedy|Romance'],
            [10, 4.5, 6.206896551724139, 'Waiting to Exhale (1995)',
             'Comedy|Drama|Romance']]
    columns = ['rating', 'users_seen', 'f_score', 'title', 'genres']
    expected_results = pd.DataFrame(data, columns=columns)
    assert not (results != expected_results).sum().sum()


@pytest.mark.usefixtures('spark_context')
def test_recommendations_pipe(spark_context):
    test_input1 = [Row(movieId=1, title='Film 1', genres='Adventure|Animation|Children|Comedy|Fantasy'),
                   Row(movieId=2, title='Film 2', genres='Adventure|Children|Fantasy'),
                   Row(movieId=3, title='Film 3', genres='Comedy|Romance'),
                   Row(movieId=4, title='Film 4', genres='Comedy|Drama|Romance'),
                   Row(movieId=5, title='Film 5', genres='Comedy')]

    test_input2 = [Row(userId=1, movieId=1, rating=3.5, timestamp=1112486027),
                   Row(userId=1, movieId=2, rating=3.5, timestamp=1112484676),
                   Row(userId=1, movieId=3, rating=5., timestamp=1112484819),
                   Row(userId=2, movieId=1, rating=3.5, timestamp=1112484727),
                   Row(userId=2, movieId=2, rating=3.5, timestamp=1112484580),
                   Row(userId=2, movieId=4, rating=4., timestamp=1112484580)]

    input_df1 = spark_context.parallelize(test_input1, 2).toDF()
    input_df2 = spark_context.parallelize(test_input2, 2).toDF()
    titles = [('film 1', 5), ('film 2', 5)]
    results = spark_functions.recommendations_pipe(spark_context, input_df1, input_df2, titles).toPandas()
    data = [[5.0, 1, 1.6666666666666667, 'Film 3', 'Comedy|Romance'],
            [4.0, 1, 1.6, 'Film 4', 'Comedy|Drama|Romance']]
    columns = ['rating', 'users_seen', 'f_score', 'title', 'genres']
    expected_results = pd.DataFrame(data, columns=columns)
    assert not (results != expected_results).sum().sum()
