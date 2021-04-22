SELECT DISTINCT *
FROM
(SELECT A.actorid as actorid1, B.actorid as actorid2
FROM playsin A, playsin B
WHERE A.actorid<>B.actorid and A.movieid=B.movieid and B.actorid IS NOT NULL
and NOT EXISTS (SELECT movieid FROM playsin WHERE actorid=A.actorid and movieid not in (SELECT movieid FROM playsin WHERE actorid=B.actorid))
ORDER BY A.actorid, B.actorid) T
