WITH RECURSIVE bacon (baconNum,baconActorId,baconMovieId) AS
(WITH baconId(actorId) AS
(SELECT actorId
FROM actors
WHERE name='Frank Bacon')
	SELECT 0, actorId, movieId
	FROM playsIn NATURAL JOIN baconId
UNION
SELECT DISTINCT (baconNum + 1), P1.actorId, P2.movieId
FROM bacon JOIN playsIn P1 ON baconMovieId=P1.movieId JOIN playsIn P2 ON P1.actorId=P2.actorId
WHERE baconNum < 5)
SELECT DISTINCT baconActorId as actorId, name
FROM bacon JOIN actors ON baconactorId=actors.actorId
UNION
SELECT actorId, name
FROM actors
WHERE name='Frank Bacon'
ORDER BY actorId;
