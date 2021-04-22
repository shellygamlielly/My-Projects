SELECT movieID, title
FROM playsIn NATURAL JOIN actors NATURAL JOIN movies
WHERE year=dyear and (genre='Documentary' or genre='Drama')
ORDER BY movieId ASC, title ASC;
