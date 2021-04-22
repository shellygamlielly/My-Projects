WITH avgRatings(actorId, avgRating) AS
(SELECT actorId, AVG(rating) as avgRating
FROM playsIn NATURAL JOIN movies
GROUP BY actorId)
SELECT actorId,name
FROM avgRatings NATURAL JOIN actors
WHERE avgRating=(SELECT MAX(avgRating) FROM avgRatings)
ORDER BY actorId;
