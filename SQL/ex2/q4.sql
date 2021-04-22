SELECT DISTINCT actorId
FROM playsIn
EXCEPT
SELECT actorId
FROM playsIn NATURAL JOIN movies
WHERE rating<=7 or rating IS NULL
ORDER BY actorId ASC;
