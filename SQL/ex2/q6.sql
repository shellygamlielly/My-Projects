SELECT DISTINCT A.movieid, E.title
FROM playsin A, playsin B, actors C, actors D, movies E
WHERE A.movieid = E.movieid and A.movieid=B.movieid and A.actorid=C.actorid and B.actorid=D.actorid and ABS(C.byear - D.byear) > 60
ORDER BY A.movieid, E.title;
