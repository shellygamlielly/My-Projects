WITH moviesWithLessThan6Actors(movieId) AS
(SELECT movieId
FROM playsIn
GROUP BY movieId
HAVING COUNT(actorId) < 6)
SELECT COUNT(*) as num
FROM (
	SELECT actorId
	FROM actors
	EXCEPT
	SELECT actorId FROM (SELECT * FROM moviesWithLessThan6Actors) T NATURAL JOIN playsIn
) actorsWhoPlayedInMoviesWith6ActorsOrMore;