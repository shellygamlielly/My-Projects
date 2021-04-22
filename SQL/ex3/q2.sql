SELECT movieId, title
FROM Movies NATURAL JOIN playsIn NATURAL JOIN Actors
GROUP BY movieId
HAVING AVG(year-byear) >= 70
ORDER BY movieId ASC
