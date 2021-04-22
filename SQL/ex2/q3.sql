SELECT actorId
FROM playsIn NATURAL JOIN movies
WHERE duration>90
INTERSECT
SELECT actorId
FROM (SELECT movieId
	FROM playsIn NATURAL JOIN actors 
	WHERE name='Charles Chaplin') MID NATURAL JOIN playsIn
ORDER BY actorId ASC;
