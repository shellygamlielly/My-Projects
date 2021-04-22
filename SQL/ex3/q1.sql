SELECT DISTINCT actorID, MAX(duration) as max, MIN(duration) as min, AVG(duration) as avg
FROM playsIn, Movies
WHERE playsIn.movieId = Movies.movieId --and duration IS NOT NULL --// rachel said it's OK to return NULL
GROUP BY actorId
ORDER BY actorId ASC;
